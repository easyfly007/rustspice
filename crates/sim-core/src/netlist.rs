#[derive(Debug, Clone)]
pub struct NetlistAst {
    pub title: Option<String>,
    pub statements: Vec<Stmt>,
    pub errors: Vec<ParseError>,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct PolySpec {
    pub degree: usize,
    pub coeffs: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Device(DeviceStmt),
    Control(ControlStmt),
    Comment(String),
    Empty,
}

#[derive(Debug, Clone)]
pub struct DeviceStmt {
    pub name: String,
    pub kind: DeviceKind,
    pub nodes: Vec<String>,
    pub model: Option<String>,
    pub control: Option<String>,
    pub value: Option<String>,
    pub params: Vec<Param>,
    pub extras: Vec<String>,
    pub poly: Option<PolySpec>,
    pub raw: String,
    pub line: usize,
    /// AC analysis magnitude (for voltage/current sources)
    pub ac_mag: Option<f64>,
    /// AC analysis phase in degrees (for voltage/current sources)
    pub ac_phase: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ControlStmt {
    pub command: String,
    pub kind: ControlKind,
    pub args: Vec<String>,
    pub params: Vec<Param>,
    pub model_name: Option<String>,
    pub model_type: Option<String>,
    pub subckt_name: Option<String>,
    pub subckt_ports: Vec<String>,
    pub raw: String,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub enum ControlKind {
    Param,
    Model,
    Subckt,
    Ends,
    Include,
    Op,
    Dc,
    Tran,
    Ac,
    Ic,
    End,
    Other,
}

#[derive(Debug, Clone)]
pub enum DeviceKind {
    R,
    C,
    L,
    V,
    I,
    D,
    M,
    E,
    G,
    F,
    H,
    X,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ElaboratedNetlist {
    pub instances: Vec<DeviceStmt>,
    pub subckt_models: Vec<ControlStmt>,
    pub control_count: usize,
    pub error_count: usize,
}

#[derive(Debug, Clone)]
pub struct SubcktDef {
    pub name: String,
    pub ports: Vec<String>,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub line: usize,
}

pub fn parse_netlist_file(path: &std::path::Path) -> NetlistAst {
    let mut errors = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let content = read_with_includes(path, &mut visited, &mut errors);
    let mut ast = parse_netlist(&content);
    ast.errors.extend(errors);
    ast
}

pub fn parse_netlist(input: &str) -> NetlistAst {
    let mut title = None;
    let mut statements = Vec::new();
    let mut errors = Vec::new();
    let mut pending_line = String::new();

    for (index, raw_line) in input.lines().enumerate() {
        let line_no = index + 1;
        let trimmed = raw_line.trim();

        if trimmed.is_empty() {
            statements.push(Stmt::Empty);
            continue;
        }

        if trimmed.starts_with('*') {
            statements.push(Stmt::Comment(trimmed.to_string()));
            continue;
        }

        if trimmed.starts_with('+') {
            if pending_line.is_empty() {
                errors.push(ParseError {
                    line: line_no,
                    message: "续行没有对应的上一行".to_string(),
                });
                continue;
            }
            pending_line.push(' ');
            pending_line.push_str(trimmed.trim_start_matches('+').trim());
            continue;
        }

        if !pending_line.is_empty() {
            parse_statement(&pending_line, line_no, &mut title, &mut statements, &mut errors);
            pending_line.clear();
        }

        pending_line = trimmed.to_string();
    }

    if !pending_line.is_empty() {
        parse_statement(&pending_line, input.lines().count(), &mut title, &mut statements, &mut errors);
    }

    NetlistAst {
        title,
        statements,
        errors,
    }
}

fn parse_statement(
    line: &str,
    line_no: usize,
    title: &mut Option<String>,
    statements: &mut Vec<Stmt>,
    errors: &mut Vec<ParseError>,
) {
    let mut iter = line.split_whitespace();
    let first = match iter.next() {
        Some(token) => token,
        None => {
            statements.push(Stmt::Empty);
            return;
        }
    };

    if first.starts_with('.') {
        let command = first.to_ascii_lowercase();
        let tokens: Vec<&str> = iter.collect();
        let (args, params) = split_args_params(&tokens);
        let kind = map_control_kind(&command);
        let mut model_name = None;
        let mut model_type = None;
        let mut subckt_name = None;
        let mut subckt_ports = Vec::new();

        if command == ".title" {
            let rest = args.join(" ");
            if !rest.is_empty() {
                *title = Some(rest);
            }
        }

        if matches!(kind, ControlKind::Model) {
            if args.len() >= 2 {
                model_name = Some(args[0].clone());
                model_type = Some(args[1].clone());
            } else {
                errors.push(ParseError {
                    line: line_no,
                    message: "model 语句缺少 name/type".to_string(),
                });
            }
        }

        if matches!(kind, ControlKind::Subckt) {
            if !args.is_empty() {
                subckt_name = Some(args[0].clone());
                if args.len() > 1 {
                    subckt_ports = args[1..].to_vec();
                }
            } else {
                errors.push(ParseError {
                    line: line_no,
                    message: "subckt 语句缺少名称".to_string(),
                });
            }
        }

        statements.push(Stmt::Control(ControlStmt {
            command,
            kind,
            args,
            params,
            model_name,
            model_type,
            subckt_name,
            subckt_ports,
            raw: line.to_string(),
            line: line_no,
        }));
        return;
    }

    let kind = match first.chars().next().unwrap_or(' ') {
        'R' | 'r' => DeviceKind::R,
        'C' | 'c' => DeviceKind::C,
        'L' | 'l' => DeviceKind::L,
        'V' | 'v' => DeviceKind::V,
        'I' | 'i' => DeviceKind::I,
        'D' | 'd' => DeviceKind::D,
        'M' | 'm' => DeviceKind::M,
        'E' | 'e' => DeviceKind::E,
        'G' | 'g' => DeviceKind::G,
        'F' | 'f' => DeviceKind::F,
        'H' | 'h' => DeviceKind::H,
        'X' | 'x' => DeviceKind::X,
        _ => DeviceKind::Unknown,
    };

    if matches!(kind, DeviceKind::Unknown) {
        errors.push(ParseError {
            line: line_no,
            message: format!("未知器件类型: {}", first),
        });
    }

    let tokens: Vec<&str> = iter.collect();
    let (args, params) = split_args_params(&tokens);
    let (nodes, model, value, extras, poly) = split_device_fields(&kind, &args);
    let control = extract_control_name(&kind, &args);
    validate_device_fields(
        first,
        &kind,
        &nodes,
        &model,
        &control,
        &value,
        &extras,
        &poly,
        line_no,
        errors,
    );

    // Extract AC parameters for V/I sources (e.g., "AC 1 0" means ac_mag=1, ac_phase=0)
    let (ac_mag, ac_phase) = extract_ac_params(&kind, &args);

    statements.push(Stmt::Device(DeviceStmt {
        name: first.to_string(),
        kind,
        nodes,
        model,
        control,
        value,
        params,
        extras,
        poly,
        raw: line.to_string(),
        line: line_no,
        ac_mag,
        ac_phase,
    }));
}

fn split_args_params(tokens: &[&str]) -> (Vec<String>, Vec<Param>) {
    let mut args = Vec::new();
    let mut params = Vec::new();

    for token in tokens {
        // 只在括号深度为 0 时按逗号分割，避免分割表达式内的逗号
        let parts: Vec<&str> = if token.contains('=') && token.contains(',') {
            split_at_top_level_commas(token)
        } else {
            vec![*token]
        };
        for raw_part in parts {
            let mut part = raw_part.trim();
            if part.is_empty() {
                continue;
            }
            if part.starts_with('(') && part.contains('=') {
                part = &part[1..];
            }
            if part.ends_with(')') && part.contains('=') {
                if let Some(eq_pos) = part.find('=') {
                    if !part[eq_pos + 1..].contains('(') {
                        part = &part[..part.len() - 1];
                    }
                }
            }
            if part.is_empty() {
                continue;
            }
            if let Some((key, value)) = part.split_once('=') {
                params.push(Param {
                    key: key.to_string(),
                    value: value.to_string(),
                });
            } else {
                args.push(part.to_string());
            }
        }
    }

    (args, params)
}

/// 只在括号深度为 0 时按逗号分割字符串
fn split_at_top_level_commas(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0;
    
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    
    if start < s.len() {
        result.push(&s[start..]);
    }
    
    result
}

fn split_device_fields(
    kind: &DeviceKind,
    args: &[String],
) -> (Vec<String>, Option<String>, Option<String>, Vec<String>, Option<PolySpec>) {
    if args.is_empty() {
        return (Vec::new(), None, None, Vec::new(), None);
    }

    let mut nodes = Vec::new();
    let mut model = None;
    let mut value = None;
    let mut extras = Vec::new();
    let mut poly = None;

    match kind {
        DeviceKind::R | DeviceKind::C | DeviceKind::L | DeviceKind::V | DeviceKind::I => {
            if args.len() >= 3 {
                nodes.extend_from_slice(&args[0..2]);
                if args[2].eq_ignore_ascii_case("dc") && args.len() >= 4 {
                    value = Some(args[3].clone());
                    if args.len() > 4 {
                        extras.extend_from_slice(&args[4..]);
                    }
                } else if matches!(kind, DeviceKind::V | DeviceKind::I) && is_waveform_token(&args[2]) {
                    extras.extend_from_slice(&args[2..]);
                } else {
                    value = Some(args[2].clone());
                    if args.len() > 3 {
                        extras.extend_from_slice(&args[3..]);
                    }
                }
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::D => {
            if args.len() >= 3 {
                nodes.extend_from_slice(&args[0..2]);
                model = Some(args[2].clone());
                if args.len() > 3 {
                    extras.extend_from_slice(&args[3..]);
                }
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::M => {
            if args.len() >= 5 {
                nodes.extend_from_slice(&args[0..4]);
                model = Some(args[4].clone());
                if args.len() > 5 {
                    extras.extend_from_slice(&args[5..]);
                }
            } else if args.len() == 4 {
                nodes.extend_from_slice(&args[0..3]);
                nodes.push("0".to_string());
                model = Some(args[3].clone());
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::E | DeviceKind::G => {
            // 检查是否有 POLY 语法
            let poly_idx = args.iter().position(|a| is_poly_token(a));
            if let Some(idx) = poly_idx {
                // POLY 语法: E1 out 0 POLY(n) ctrl1+ ctrl1- ... coeffs
                // 或者: E1 out 0 in 0 POLY(n) ctrl2+ ctrl2- ... coeffs
                // 输出节点总是前 2 个
                if args.len() >= 2 {
                    nodes.extend_from_slice(&args[0..2]);
                }
                // POLY 之前的额外节点是第一组控制节点
                if idx > 2 {
                    extras.extend_from_slice(&args[2..idx]);
                }
                // POLY 及之后的内容
                extras.extend_from_slice(&args[idx..]);
            } else if args.len() >= 5 {
                // 普通语法: E1 out 0 in 0 gain
                nodes.extend_from_slice(&args[0..4]);
                value = Some(args[4].clone());
                if args.len() > 5 {
                    extras.extend_from_slice(&args[5..]);
                }
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::F | DeviceKind::H => {
            // 检查是否有 POLY 语法
            let poly_idx = args.iter().position(|a| is_poly_token(a));
            if let Some(idx) = poly_idx {
                // POLY 语法: F1 out 0 POLY(n) V1 V2 ... coeffs
                if args.len() >= 2 {
                    nodes.extend_from_slice(&args[0..2]);
                }
                // POLY 及之后的内容
                extras.extend_from_slice(&args[idx..]);
            } else if args.len() >= 4 {
                // 普通语法: F1 out 0 Vctrl gain
                // control 由 extract_control_name 提取
                nodes.extend_from_slice(&args[0..2]);
                value = Some(args[3].clone());
                if args.len() > 4 {
                    extras.extend_from_slice(&args[4..]);
                }
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::X => {
            if args.len() >= 2 {
                nodes.extend_from_slice(&args[0..args.len() - 1]);
                model = Some(args[args.len() - 1].clone());
            } else {
                nodes.extend_from_slice(args);
            }
        }
        DeviceKind::Unknown => {
            nodes.extend_from_slice(args);
        }
    }

    if matches!(kind, DeviceKind::E | DeviceKind::G | DeviceKind::F | DeviceKind::H) {
        let (poly_spec, remaining) = parse_poly(&extras);
        poly = poly_spec;
        extras = remaining;
        if poly.is_some() {
            value = None;
        }
    }

    (nodes, model, value, extras, poly)
}

fn is_poly_token(token: &str) -> bool {
    token.to_ascii_uppercase().starts_with("POLY")
}

fn is_waveform_token(token: &str) -> bool {
    let upper = token.to_ascii_uppercase();
    upper == "AC"
        || upper == "SIN"
        || upper == "PULSE"
        || upper == "EXP"
        || upper == "SFFM"
        || upper == "PWL"
        || upper.starts_with("SIN(")
        || upper.starts_with("PULSE(")
        || upper.starts_with("EXP(")
        || upper.starts_with("SFFM(")
        || upper.starts_with("PWL(")
        || upper.starts_with("AC(")
}

fn map_control_kind(command: &str) -> ControlKind {
    match command {
        ".param" => ControlKind::Param,
        ".model" => ControlKind::Model,
        ".subckt" => ControlKind::Subckt,
        ".ends" => ControlKind::Ends,
        ".include" => ControlKind::Include,
        ".op" => ControlKind::Op,
        ".dc" => ControlKind::Dc,
        ".tran" => ControlKind::Tran,
        ".ac" => ControlKind::Ac,
        ".ic" => ControlKind::Ic,
        ".end" => ControlKind::End,
        _ => ControlKind::Other,
    }
}

fn validate_device_fields(
    name: &str,
    kind: &DeviceKind,
    nodes: &[String],
    model: &Option<String>,
    control: &Option<String>,
    value: &Option<String>,
    extras: &[String],
    poly: &Option<PolySpec>,
    line_no: usize,
    errors: &mut Vec<ParseError>,
) {
    if matches!(kind, DeviceKind::Unknown) {
        return;
    }

    if nodes.is_empty() {
        errors.push(ParseError {
            line: line_no,
            message: format!("器件缺少节点定义: {} {}", name, format_fields(nodes, model, control, value, extras, poly)),
        });
        return;
    }

    match kind {
        DeviceKind::R
        | DeviceKind::C
        | DeviceKind::L
        | DeviceKind::V
        | DeviceKind::I => {
            if nodes.len() != 2 {
                errors.push(ParseError {
                    line: line_no,
                    message: format!(
                        "{} 需要 2 个节点，当前={} {}",
                        name,
                        nodes.len(),
                        format_fields(nodes, model, control, value, extras, poly)
                    ),
                });
            }
            if value.is_none() && !(matches!(kind, DeviceKind::V | DeviceKind::I) && has_waveform(extras)) {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少数值 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if matches!(kind, DeviceKind::R | DeviceKind::C | DeviceKind::L) && !extras.is_empty() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 存在多余字段 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if matches!(kind, DeviceKind::V | DeviceKind::I)
                && !extras.is_empty()
                && !has_waveform(extras)
            {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 存在多余字段 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if matches!(kind, DeviceKind::V | DeviceKind::I) {
                if let Some(wave) = waveform_keyword(extras) {
                    if extras.len() == 1 {
                        errors.push(ParseError {
                            line: line_no,
                            message: format!("{} 波形 {} 缺少参数 {}", name, wave, format_fields(nodes, model, control, value, extras, poly)),
                        });
                    }
                }
            }
        }
        DeviceKind::D => {
            if nodes.len() != 2 {
                errors.push(ParseError {
                    line: line_no,
                    message: format!(
                        "{} 需要 2 个节点，当前={} {}",
                        name,
                        nodes.len(),
                        format_fields(nodes, model, control, value, extras, poly)
                    ),
                });
            }
            if model.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少模型名 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
        }
        DeviceKind::M => {
            if nodes.len() < 4 {
                errors.push(ParseError {
                    line: line_no,
                    message: format!(
                        "{} 需要至少 4 个节点，当前={} {}",
                        name,
                        nodes.len(),
                        format_fields(nodes, model, control, value, extras, poly)
                    ),
                });
            }
            if model.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少模型名 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
        }
        DeviceKind::E | DeviceKind::G => {
            if poly.is_some() {
                // POLY 语法: 输出节点 2 个
                if nodes.len() != 2 {
                    errors.push(ParseError {
                        line: line_no,
                        message: format!(
                            "{} POLY 需要 2 个输出节点，当前={} {}",
                            name,
                            nodes.len(),
                            format_fields(nodes, model, control, value, extras, poly)
                        ),
                    });
                }
            } else {
                // 普通语法: 4 个节点 (out+ out- in+ in-)
                if nodes.len() != 4 {
                    errors.push(ParseError {
                        line: line_no,
                        message: format!(
                            "{} 需要 4 个节点，当前={} {}",
                            name,
                            nodes.len(),
                            format_fields(nodes, model, control, value, extras, poly)
                        ),
                    });
                }
            }
            if value.is_none() && poly.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少增益值 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if poly.is_none() && !extras.is_empty() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 存在多余字段 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if let Some(spec) = poly {
                // POLY 后面的内容 (coeffs): 控制节点 + 系数
                // 控制节点数量 = degree * 2 (每对是 v+ v-)
                // POLY 之前的内容（如 in 0）是可选的兼容语法，不计入控制节点
                let expected_controls = spec.degree * 2;
                if spec.coeffs.len() < expected_controls {
                    errors.push(ParseError {
                        line: line_no,
                        message: format!(
                            "{} POLY 控制节点数量不足，期望={} 实际={} {}",
                            name,
                            expected_controls,
                            spec.coeffs.len(),
                            format_fields(nodes, model, control, value, extras, poly)
                        ),
                    });
                } else {
                    // 系数是控制节点之后的部分
                    let actual_coeffs = spec.coeffs.len() - expected_controls;
                    if actual_coeffs < spec.degree + 1 {
                        errors.push(ParseError {
                            line: line_no,
                            message: format!(
                                "{} POLY 系数数量不足，期望>={} 实际={} {}",
                                name,
                                spec.degree + 1,
                                actual_coeffs,
                                format_fields(nodes, model, control, value, extras, poly)
                            ),
                        });
                    }
                }
            }
        }
        DeviceKind::F | DeviceKind::H => {
            if nodes.len() != 2 {
                errors.push(ParseError {
                    line: line_no,
                    message: format!(
                        "{} 需要 2 个节点，当前={} {}",
                        name,
                        nodes.len(),
                        format_fields(nodes, model, control, value, extras, poly)
                    ),
                });
            }
            if control.is_none() && poly.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少控制源 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if value.is_none() && poly.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少增益值 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if poly.is_none() && !extras.is_empty() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 存在多余字段 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if let Some(spec) = poly {
                // POLY 后面的内容: 控制源名称 + 系数
                // 控制源数量 = degree
                let expected_controls = spec.degree;
                if spec.coeffs.len() < expected_controls {
                    errors.push(ParseError {
                        line: line_no,
                        message: format!(
                            "{} POLY 控制源数量不足，期望={} 实际={} {}",
                            name,
                            expected_controls,
                            spec.coeffs.len(),
                            format_fields(nodes, model, control, value, extras, poly)
                        ),
                    });
                } else {
                    // 系数是控制源之后的部分
                    let actual_coeffs = spec.coeffs.len() - expected_controls;
                    if actual_coeffs < spec.degree + 1 {
                        errors.push(ParseError {
                            line: line_no,
                            message: format!(
                                "{} POLY 系数数量不足，期望>={} 实际={} {}",
                                name,
                                spec.degree + 1,
                                actual_coeffs,
                                format_fields(nodes, model, control, value, extras, poly)
                            ),
                        });
                    }
                }
            }
        }
        DeviceKind::X => {
            if nodes.is_empty() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少节点 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
            if model.is_none() {
                errors.push(ParseError {
                    line: line_no,
                    message: format!("{} 缺少子电路名 {}", name, format_fields(nodes, model, control, value, extras, poly)),
                });
            }
        }
        DeviceKind::Unknown => {}
    }
}

pub fn elaborate_netlist(ast: &NetlistAst) -> ElaboratedNetlist {
    let mut errors = ast.errors.clone();
    let (top_level, subckts, subckt_errors) = extract_subckts(&ast.statements);
    errors.extend(subckt_errors);

    let param_table = build_param_table(&top_level);
    let subckt_map = build_subckt_map(&subckts);
    let mut instances = Vec::new();
    let mut subckt_models = Vec::new();
    let mut control_count = 0;

    for stmt in top_level {
        match stmt {
            Stmt::Device(device) => {
                if matches!(device.kind, DeviceKind::X) {
                    if let Some(subckt_name) = device.model.as_deref() {
                        if let Some(def) = subckt_map.get(subckt_name) {
                            let body_params = collect_params_from_body(&def.body);
                            let local_params = build_local_param_table(
                                def,
                                &device,
                                &body_params,
                                &std::collections::HashMap::new(),
                                &param_table,
                            );
                            let expanded = expand_subckt_instance_recursive(
                                &device,
                                def,
                                &subckt_map,
                                &local_params,
                                &param_table,
                                &mut errors,
                                &mut subckt_models,
                            );
                            instances.extend(expanded);
                            continue;
                        }
                    }
                    errors.push(ParseError {
                        line: device.line,
                        message: format!("子电路未定义: {:?}", device.model),
                    });
                    let mut fallback = device.clone();
                    apply_params_to_device_scoped(&param_table, &std::collections::HashMap::new(), &mut fallback);
                    instances.push(fallback);
                } else {
                    let mut inst = device.clone();
                    apply_params_to_device_scoped(&param_table, &std::collections::HashMap::new(), &mut inst);
                    instances.push(inst);
                }
            }
            Stmt::Control(_) => {
                control_count += 1;
            }
            _ => {}
        }
    }

    ElaboratedNetlist {
        instances,
        subckt_models,
        control_count,
        error_count: errors.len(),
    }
}

pub fn build_circuit(ast: &NetlistAst, elab: &ElaboratedNetlist) -> crate::circuit::Circuit {
    use crate::circuit::{AnalysisCmd, Circuit, DeviceKind as CircuitDeviceKind, Instance, Model, PolySpec as CircuitPolySpec};
    use std::collections::HashMap;

    let mut circuit = Circuit::new();

    // Process top-level statements from AST
    for stmt in &ast.statements {
        if let Stmt::Control(ctrl) = stmt {
            match ctrl.kind {
                ControlKind::Model => {
                    if let (Some(name), Some(model_type)) = (&ctrl.model_name, &ctrl.model_type) {
                        let mut params = HashMap::new();
                        for param in &ctrl.params {
                            params.insert(param.key.to_ascii_lowercase(), param.value.clone());
                        }
                        let name_norm = name.to_ascii_lowercase();
                        let model_type_norm = model_type.to_ascii_lowercase();
                        circuit.models.insert(Model {
                            name: name_norm,
                            model_type: model_type_norm,
                            params,
                        });
                    }
                }
                ControlKind::Op => {
                    circuit.analysis.push(AnalysisCmd::Op);
                }
                ControlKind::Dc => {
                    if ctrl.args.len() >= 4 {
                        let source = ctrl.args[0].clone();
                        let start = parse_number_with_suffix(&ctrl.args[1])
                            .or_else(|| ctrl.args[1].parse().ok())
                            .unwrap_or(0.0);
                        let stop = parse_number_with_suffix(&ctrl.args[2])
                            .or_else(|| ctrl.args[2].parse().ok())
                            .unwrap_or(0.0);
                        let step = parse_number_with_suffix(&ctrl.args[3])
                            .or_else(|| ctrl.args[3].parse().ok())
                            .unwrap_or(0.0);
                        circuit.analysis.push(AnalysisCmd::Dc {
                            source,
                            start,
                            stop,
                            step,
                        });
                    }
                }
                ControlKind::Tran => {
                    if ctrl.args.len() >= 2 {
                        let tstep = parse_number_with_suffix(&ctrl.args[0])
                            .or_else(|| ctrl.args[0].parse().ok())
                            .unwrap_or(0.0);
                        let tstop = parse_number_with_suffix(&ctrl.args[1])
                            .or_else(|| ctrl.args[1].parse().ok())
                            .unwrap_or(0.0);
                        let tstart = ctrl
                            .args
                            .get(2)
                            .and_then(|v| parse_number_with_suffix(v).or_else(|| v.parse().ok()))
                            .unwrap_or(0.0);
                        let tmax = ctrl
                            .args
                            .get(3)
                            .and_then(|v| parse_number_with_suffix(v).or_else(|| v.parse().ok()))
                            .unwrap_or(tstop);
                        circuit.analysis.push(AnalysisCmd::Tran {
                            tstep,
                            tstop,
                            tstart,
                            tmax,
                        });
                    }
                }
                ControlKind::Ac => {
                    // .ac dec|oct|lin <points> <fstart> <fstop>
                    if ctrl.args.len() >= 4 {
                        let sweep_type = match ctrl.args[0].to_ascii_lowercase().as_str() {
                            "dec" => crate::circuit::AcSweepType::Dec,
                            "oct" => crate::circuit::AcSweepType::Oct,
                            "lin" => crate::circuit::AcSweepType::Lin,
                            _ => crate::circuit::AcSweepType::Dec, // default to decade
                        };
                        let points = ctrl.args[1].parse().unwrap_or(10);
                        let fstart = parse_number_with_suffix(&ctrl.args[2])
                            .or_else(|| ctrl.args[2].parse().ok())
                            .unwrap_or(1.0);
                        let fstop = parse_number_with_suffix(&ctrl.args[3])
                            .or_else(|| ctrl.args[3].parse().ok())
                            .unwrap_or(1e6);
                        circuit.analysis.push(AnalysisCmd::Ac {
                            sweep_type,
                            points,
                            fstart,
                            fstop,
                        });
                    }
                }
                ControlKind::Ic => {
                    // Parse .ic v(node1)=value v(node2)=value ...
                    // The parser splits "v(node)=value" into params with key="v(node)" and value="value"
                    for param in &ctrl.params {
                        if let Some(node_name) = parse_ic_node_key(&param.key) {
                            if let Some(value) = parse_number_with_suffix(&param.value) {
                                let node_id = circuit.nodes.ensure_node(&node_name);
                                circuit.initial_conditions.insert(node_id, value);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Process .model statements extracted from subcircuits during elaboration
    for ctrl in &elab.subckt_models {
        if let (Some(name), Some(model_type)) = (&ctrl.model_name, &ctrl.model_type) {
            let mut params = HashMap::new();
            for param in &ctrl.params {
                params.insert(param.key.to_ascii_lowercase(), param.value.clone());
            }
            let name_norm = name.to_ascii_lowercase();
            let model_type_norm = model_type.to_ascii_lowercase();
            circuit.models.insert(Model {
                name: name_norm,
                model_type: model_type_norm,
                params,
            });
        }
    }

    for device in &elab.instances {
        let kind = match device.kind {
            DeviceKind::R => Some(CircuitDeviceKind::R),
            DeviceKind::C => Some(CircuitDeviceKind::C),
            DeviceKind::L => Some(CircuitDeviceKind::L),
            DeviceKind::V => Some(CircuitDeviceKind::V),
            DeviceKind::I => Some(CircuitDeviceKind::I),
            DeviceKind::D => Some(CircuitDeviceKind::D),
            DeviceKind::M => Some(CircuitDeviceKind::M),
            DeviceKind::E => Some(CircuitDeviceKind::E),
            DeviceKind::G => Some(CircuitDeviceKind::G),
            DeviceKind::F => Some(CircuitDeviceKind::F),
            DeviceKind::H => Some(CircuitDeviceKind::H),
            DeviceKind::X => Some(CircuitDeviceKind::X),
            DeviceKind::Unknown => None,
        };
        let Some(kind) = kind else {
            continue;
        };

        let nodes = device
            .nodes
            .iter()
            .map(|name| circuit.nodes.ensure_node(name))
            .collect::<Vec<_>>();

        let model = device.model.as_ref().and_then(|name| {
            let key = name.to_ascii_lowercase();
            circuit.models.name_to_id.get(&key).copied()
        });

        let mut params = HashMap::new();
        if let Some(model_id) = model {
            if let Some(model_def) = circuit.models.models.get(model_id.0) {
                params.extend(model_def.params.clone());
            }
        }
        for param in &device.params {
            params.insert(param.key.to_ascii_lowercase(), param.value.clone());
        }

        // Build POLY specification if present
        let poly = if let Some(ref poly_spec) = device.poly {
            let is_voltage_controlled = matches!(kind, CircuitDeviceKind::E | CircuitDeviceKind::G);
            let is_current_controlled = matches!(kind, CircuitDeviceKind::F | CircuitDeviceKind::H);

            if is_voltage_controlled {
                // For E/G: first 2*n items in coeffs are control node names, rest are coefficients
                let num_control_nodes = poly_spec.degree * 2;
                let mut control_nodes = Vec::new();
                let mut coeffs = Vec::new();

                for (i, item) in poly_spec.coeffs.iter().enumerate() {
                    if i < num_control_nodes {
                        // This is a control node name - ensure it exists in the circuit
                        let node_id = circuit.nodes.ensure_node(item);
                        // Pair up: (pos, neg) for each control input
                        if i % 2 == 1 {
                            let pos_id = circuit.nodes.ensure_node(&poly_spec.coeffs[i - 1]);
                            control_nodes.push((pos_id.0, node_id.0));
                        }
                    } else {
                        // This is a coefficient - parse as number
                        if let Some(val) = parse_number_with_suffix(item).or_else(|| item.parse().ok()) {
                            coeffs.push(val);
                        }
                    }
                }

                Some(CircuitPolySpec {
                    degree: poly_spec.degree,
                    coeffs,
                    control_nodes,
                    control_sources: Vec::new(),
                })
            } else if is_current_controlled {
                // For F/H: first n items in coeffs are control source names, rest are coefficients
                let num_control_sources = poly_spec.degree;
                let mut control_sources = Vec::new();
                let mut coeffs = Vec::new();

                for (i, item) in poly_spec.coeffs.iter().enumerate() {
                    if i < num_control_sources {
                        // This is a control source name
                        control_sources.push(item.clone());
                    } else {
                        // This is a coefficient - parse as number
                        if let Some(val) = parse_number_with_suffix(item).or_else(|| item.parse().ok()) {
                            coeffs.push(val);
                        }
                    }
                }

                Some(CircuitPolySpec {
                    degree: poly_spec.degree,
                    coeffs,
                    control_nodes: Vec::new(),
                    control_sources,
                })
            } else {
                None
            }
        } else {
            None
        };

        circuit.instances.insert(Instance {
            name: device.name.clone(),
            kind,
            nodes,
            model,
            params,
            value: device.value.clone(),
            control: device.control.clone(),
            ac_mag: device.ac_mag,
            ac_phase: device.ac_phase,
            poly,
        });
    }

    if circuit.analysis.is_empty() {
        circuit.analysis.push(AnalysisCmd::Op);
    }

    circuit
}

fn read_with_includes(
    path: &std::path::Path,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
    errors: &mut Vec<ParseError>,
) -> String {
    if !visited.insert(path.to_path_buf()) {
        errors.push(ParseError {
            line: 0,
            message: format!("include 循环引用: {}", path.display()),
        });
        return String::new();
    }

    let content = std::fs::read_to_string(path).unwrap_or_else(|_| {
        errors.push(ParseError {
            line: 0,
            message: format!("无法读取文件: {}", path.display()),
        });
        String::new()
    });

    let mut out = String::new();
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.to_ascii_lowercase().starts_with(".include") {
            let include_path = trimmed
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .trim_matches('"');
            if include_path.is_empty() {
                errors.push(ParseError {
                    line: 0,
                    message: format!("include 语句缺少路径: {}", path.display()),
                });
                continue;
            }
            let include_file = base_dir.join(include_path);
            let nested = read_with_includes(&include_file, visited, errors);
            out.push_str(&nested);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

fn build_param_table(statements: &[Stmt]) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();
    for stmt in statements {
        if let Stmt::Control(ctrl) = stmt {
            if matches!(ctrl.kind, ControlKind::Param) {
                for param in &ctrl.params {
                    let key = param.key.to_ascii_lowercase();
                    let value = eval_expression(&params, &param.value)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| param.value.clone());
                    params.insert(key, value);
                }
            }
        }
    }
    params
}

fn build_local_param_table(
    def: &SubcktDef,
    instance: &DeviceStmt,
    body_params: &[Param],
    parent: &std::collections::HashMap<String, String>,
    global: &std::collections::HashMap<String, String>,
) -> std::collections::HashMap<String, String> {
    let mut params = std::collections::HashMap::new();
    for param in &def.params {
        let key = param.key.to_ascii_lowercase();
        let value = eval_expression_scoped(&params, parent, global, &param.value)
            .map(|v| v.to_string())
            .unwrap_or_else(|| param.value.clone());
        params.insert(key, value);
    }
    for param in body_params {
        let key = param.key.to_ascii_lowercase();
        let value = eval_expression_scoped(&params, parent, global, &param.value)
            .map(|v| v.to_string())
            .unwrap_or_else(|| param.value.clone());
        params.insert(key, value);
    }
    for param in &instance.params {
        let key = param.key.to_ascii_lowercase();
        let value = eval_expression_scoped(&params, parent, global, &param.value)
            .map(|v| v.to_string())
            .unwrap_or_else(|| param.value.clone());
        params.insert(key, value);
    }
    params
}

fn apply_params_to_device_scoped(
    global: &std::collections::HashMap<String, String>,
    local: &std::collections::HashMap<String, String>,
    device: &mut DeviceStmt,
) {
    if let Some(value) = device.value.clone() {
        if let Some(replaced) = resolve_param_scoped(local, global, &value) {
            device.value = Some(replaced);
        }
    }
    if let Some(model) = device.model.clone() {
        if let Some(replaced) = resolve_param_scoped(local, global, &model) {
            device.model = Some(replaced);
        }
    }
    for param in &mut device.params {
        if let Some(replaced) = resolve_param_scoped(local, global, &param.value) {
            param.value = replaced;
        }
    }
}

fn resolve_param_scoped(
    local: &std::collections::HashMap<String, String>,
    global: &std::collections::HashMap<String, String>,
    token: &str,
) -> Option<String> {
    let key = token.to_ascii_lowercase();
    local
        .get(&key)
        .cloned()
        .or_else(|| global.get(&key).cloned())
        .or_else(|| eval_expression_scoped(local, &std::collections::HashMap::new(), global, token).map(|v| v.to_string()))
}

fn extract_subckts(statements: &[Stmt]) -> (Vec<Stmt>, Vec<SubcktDef>, Vec<ParseError>) {
    let mut top_level = Vec::new();
    let mut subckts = Vec::new();
    let mut errors = Vec::new();
    let mut idx = 0;

    while idx < statements.len() {
        match &statements[idx] {
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Subckt) => {
                let name = ctrl.subckt_name.clone().unwrap_or_else(|| "unknown".to_string());
                let ports = ctrl.subckt_ports.clone();
                let params = ctrl.params.clone();
                let line = ctrl.line;
                idx += 1;
                let mut body = Vec::new();
                let mut found_ends = false;

                while idx < statements.len() {
                    match &statements[idx] {
                        Stmt::Control(end_ctrl) if matches!(end_ctrl.kind, ControlKind::Ends) => {
                            found_ends = true;
                            idx += 1;
                            break;
                        }
                        stmt => {
                            body.push(stmt.clone());
                            idx += 1;
                        }
                    }
                }

                if !found_ends {
                    errors.push(ParseError {
                        line,
                        message: format!("subckt {} 缺少 .ends", name),
                    });
                }

                subckts.push(SubcktDef {
                    name,
                    ports,
                    params,
                    body,
                    line,
                });
            }
            stmt => {
                top_level.push(stmt.clone());
                idx += 1;
            }
        }
    }

    (top_level, subckts, errors)
}

fn collect_params_from_body(body: &[Stmt]) -> Vec<Param> {
    let mut params = Vec::new();
    let mut depth = 0usize;
    for stmt in body {
        match stmt {
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Subckt) => {
                depth += 1;
            }
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Ends) => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Param) && depth == 0 => {
                params.extend(ctrl.params.clone());
            }
            _ => {}
        }
    }
    params
}

fn map_subckt_node(
    instance: &DeviceStmt,
    port_map: &std::collections::HashMap<String, String>,
    node: &str,
) -> String {
    // SPICE 节点名大小写不敏感，使用小写查找
    if let Some(mapped) = port_map.get(&node.to_ascii_lowercase()) {
        return mapped.clone();
    }
    format!("{}:{}", instance.name, node)
}

fn extract_control_name(kind: &DeviceKind, args: &[String]) -> Option<String> {
    match kind {
        DeviceKind::F | DeviceKind::H => match args.get(2) {
            Some(token) if is_poly_token(token) => None,
            Some(token) => Some(token.clone()),
            None => None,
        },
        _ => None,
    }
}

/// Extract AC magnitude and phase from V/I source arguments.
/// Syntax: `V1 in 0 DC 1 AC <mag> [<phase>]`
fn extract_ac_params(kind: &DeviceKind, args: &[String]) -> (Option<f64>, Option<f64>) {
    if !matches!(kind, DeviceKind::V | DeviceKind::I) {
        return (None, None);
    }

    // Find the AC keyword and extract following values
    let mut ac_idx = None;
    for (i, arg) in args.iter().enumerate() {
        if arg.to_ascii_uppercase() == "AC" {
            ac_idx = Some(i);
            break;
        }
    }

    let Some(idx) = ac_idx else {
        return (None, None);
    };

    let mag = args.get(idx + 1).and_then(|s| parse_number_with_suffix(s).or_else(|| s.parse().ok()));
    let phase = args.get(idx + 2).and_then(|s| parse_number_with_suffix(s).or_else(|| s.parse().ok()));

    (mag, phase)
}

fn has_waveform(extras: &[String]) -> bool {
    extras.iter().any(|token| is_waveform_token(token))
}

fn waveform_keyword(extras: &[String]) -> Option<String> {
    let token = extras.first()?;
    let upper = token.to_ascii_uppercase();
    match upper.as_str() {
        "AC" | "SIN" | "PULSE" | "EXP" | "SFFM" | "PWL" => Some(upper),
        _ => None,
    }
}

fn format_fields(
    nodes: &[String],
    model: &Option<String>,
    control: &Option<String>,
    value: &Option<String>,
    extras: &[String],
    poly: &Option<PolySpec>,
) -> String {
    let mut parts = Vec::new();
    if !nodes.is_empty() {
        parts.push(format!("nodes={}", nodes.join(",")));
    }
    if let Some(m) = model {
        parts.push(format!("model={}", m));
    }
    if let Some(c) = control {
        parts.push(format!("ctrl={}", c));
    }
    if let Some(v) = value {
        parts.push(format!("value={}", v));
    }
    if !extras.is_empty() {
        parts.push(format!("extras={}", extras.join(",")));
    }
    if let Some(spec) = poly {
        parts.push(format!("poly=deg{} coeffs={}", spec.degree, spec.coeffs.len()));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("[{}]", parts.join(" "))
    }
}

fn parse_poly(tokens: &[String]) -> (Option<PolySpec>, Vec<String>) {
    for (idx, token) in tokens.iter().enumerate() {
        let upper = token.to_ascii_uppercase();
        if upper.starts_with("POLY") {
            let mut degree_token = None;
            let mut skip = 1;

            if upper.starts_with("POLY(") && upper.ends_with(')') {
                degree_token = Some(token.trim_start_matches("POLY(").trim_end_matches(')'));
            } else if idx + 1 < tokens.len() {
                let next = tokens[idx + 1].trim();
                if next.starts_with('(') && next.ends_with(')') {
                    degree_token = Some(next.trim_start_matches('(').trim_end_matches(')'));
                    skip = 2;
                } else if next.chars().all(|c| c.is_ascii_digit()) {
                    degree_token = Some(next);
                    skip = 2;
                }
            }

            let degree = degree_token
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            let coeffs = tokens[idx + skip..].to_vec();
            let mut remaining = Vec::new();
            remaining.extend_from_slice(&tokens[..idx]);
            return (Some(PolySpec { degree, coeffs }), remaining);
        }
    }
    (None, tokens.to_vec())
}

fn eval_expression(
    params: &std::collections::HashMap<String, String>,
    expr: &str,
) -> Option<f64> {
    eval_expression_scoped(params, params, params, expr)
}

fn eval_expression_scoped(
    local: &std::collections::HashMap<String, String>,
    parent: &std::collections::HashMap<String, String>,
    global: &std::collections::HashMap<String, String>,
    expr: &str,
) -> Option<f64> {
    let tokens = tokenize_expr(expr);
    if tokens.is_empty() {
        return None;
    }
    let rpn = to_rpn(tokens)?;
    eval_rpn(&rpn, local, parent, global)
}

#[derive(Debug, Clone)]
enum ExprToken {
    Number(f64),
    Ident(String),
    Op(char),
    Comma,
    LParen,
    RParen,
    Func { name: String, argc: usize },
}

fn tokenize_expr(expr: &str) -> Vec<ExprToken> {
    let mut tokens = Vec::new();
    let mut buf = String::new();

    let push_buf = |buf: &mut String, tokens: &mut Vec<ExprToken>| {
        if buf.is_empty() {
            return;
        }
        if let Some(num) = parse_number_with_suffix(buf) {
            tokens.push(ExprToken::Number(num));
        } else {
            tokens.push(ExprToken::Ident(buf.to_string()));
        }
        buf.clear();
    };

    let mut chars = expr.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            push_buf(&mut buf, &mut tokens);
            continue;
        }
        match ch {
            '+' | '-' | '*' | '/' | '^' => {
                push_buf(&mut buf, &mut tokens);
                tokens.push(ExprToken::Op(ch));
            }
            ',' => {
                push_buf(&mut buf, &mut tokens);
                tokens.push(ExprToken::Comma);
            }
            '(' => {
                push_buf(&mut buf, &mut tokens);
                tokens.push(ExprToken::LParen);
            }
            ')' => {
                push_buf(&mut buf, &mut tokens);
                tokens.push(ExprToken::RParen);
            }
            _ => buf.push(ch),
        }
    }
    push_buf(&mut buf, &mut tokens);
    tokens
}

fn parse_number_with_suffix(token: &str) -> Option<f64> {
    let lower = token.to_ascii_lowercase();
    let trimmed = lower.trim();
    let (num_str, multiplier) = if trimmed.ends_with("meg") {
        (&trimmed[..trimmed.len() - 3], 1e6)
    } else {
        let (value_part, suffix) = trimmed.split_at(trimmed.len().saturating_sub(1));
        match suffix {
            "f" => (value_part, 1e-15),
            "p" => (value_part, 1e-12),
            "n" => (value_part, 1e-9),
            "u" => (value_part, 1e-6),
            "m" => (value_part, 1e-3),
            "k" => (value_part, 1e3),
            "g" => (value_part, 1e9),
            "t" => (value_part, 1e12),
            _ => (trimmed, 1.0),
        }
    };

    if let Ok(num) = num_str.parse::<f64>() {
        Some(num * multiplier)
    } else {
        None
    }
}

/// Parse an IC node key like "v(node)" or "V(NODE)"
/// Returns the node name (lowercase) if successful
fn parse_ic_node_key(key: &str) -> Option<String> {
    let key = key.trim().to_ascii_lowercase();

    // Check for v(node) format
    if !key.starts_with("v(") || !key.ends_with(')') {
        return None;
    }

    // Extract node name (between 'v(' and ')')
    let node_name = key[2..key.len() - 1].trim().to_string();
    if node_name.is_empty() {
        return None;
    }

    Some(node_name)
}

fn to_rpn(tokens: Vec<ExprToken>) -> Option<Vec<ExprToken>> {
    let mut output = Vec::new();
    let mut ops: Vec<ExprToken> = Vec::new();
    let mut arg_stack: Vec<usize> = Vec::new();
    let mut idx = 0;
    let mut prev_was_value = false;

    while idx < tokens.len() {
        let token = tokens[idx].clone();
        let next = tokens.get(idx + 1);
        match &token {
            ExprToken::Number(_) => output.push(token.clone()),
            ExprToken::Ident(name) => {
                if matches!(next, Some(ExprToken::LParen)) {
                    ops.push(ExprToken::Func { name: name.clone(), argc: 0 });
                } else {
                    output.push(ExprToken::Ident(name.clone()));
                }
            }
            ExprToken::Op(op) => {
                if !prev_was_value && *op == '-' {
                    output.push(ExprToken::Number(0.0));
                }
                while let Some(top) = ops.last() {
                    match top {
                        ExprToken::Op(top_op) if precedence(*top_op) >= precedence(*op) => {
                            output.push(ops.pop().unwrap());
                        }
                        _ => break,
                    }
                }
                ops.push(ExprToken::Op(*op));
            }
            ExprToken::Comma => {
                while let Some(top) = ops.last() {
                    if matches!(top, ExprToken::LParen) {
                        break;
                    }
                    output.push(ops.pop().unwrap());
                }
                if let Some(top) = arg_stack.last_mut() {
                    *top += 1;
                }
            }
            ExprToken::LParen => {
                if matches!(ops.last(), Some(ExprToken::Func { .. })) {
                    arg_stack.push(0);
                }
                ops.push(ExprToken::LParen);
            }
            ExprToken::RParen => {
                while let Some(top) = ops.pop() {
                    if matches!(top, ExprToken::LParen) {
                        break;
                    }
                    output.push(top);
                }
                if let Some(ExprToken::Func { name, .. }) = ops.last() {
                    let argc = arg_stack.pop().unwrap_or(0) + 1;
                    let func = ExprToken::Func {
                        name: name.clone(),
                        argc,
                    };
                    ops.pop();
                    output.push(func);
                }
            }
            ExprToken::Func { .. } => {}
        }
        prev_was_value = matches!(
            token,
            ExprToken::Number(_) | ExprToken::Ident(_) | ExprToken::RParen
        );
        idx += 1;
    }

    while let Some(op) = ops.pop() {
        if matches!(op, ExprToken::LParen | ExprToken::RParen) {
            return None;
        }
        output.push(op);
    }

    Some(output)
}

fn precedence(op: char) -> u8 {
    match op {
        '+' | '-' => 1,
        '*' | '/' => 2,
        '^' => 3,
        _ => 0,
    }
}

fn eval_rpn(
    rpn: &[ExprToken],
    local: &std::collections::HashMap<String, String>,
    parent: &std::collections::HashMap<String, String>,
    global: &std::collections::HashMap<String, String>,
) -> Option<f64> {
    let mut stack: Vec<f64> = Vec::new();
    for token in rpn {
        match token {
            ExprToken::Number(num) => stack.push(*num),
            ExprToken::Ident(name) => {
                let key = name.to_ascii_lowercase();
                let value = local
                    .get(&key)
                    .or_else(|| parent.get(&key))
                    .or_else(|| global.get(&key))
                    .and_then(|val| parse_number_with_suffix(val).or_else(|| val.parse().ok()))?;
                stack.push(value);
            }
            ExprToken::Op(op) => {
                let b = stack.pop()?;
                let a = stack.pop()?;
                let value = match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    '^' => a.powf(b),
                    _ => return None,
                };
                stack.push(value);
            }
            ExprToken::Func { name, argc } => {
                let mut args = Vec::new();
                for _ in 0..*argc {
                    args.push(stack.pop()?);
                }
                args.reverse();
                let value = eval_function(name, &args)?;
                stack.push(value);
            }
            ExprToken::Comma | ExprToken::LParen | ExprToken::RParen => return None,
        }
    }
    if stack.len() == 1 {
        Some(stack[0])
    } else {
        None
    }
}

fn eval_function(name: &str, args: &[f64]) -> Option<f64> {
    match name.to_ascii_lowercase().as_str() {
        "max" if args.len() == 2 => Some(args[0].max(args[1])),
        "min" if args.len() == 2 => Some(args[0].min(args[1])),
        "abs" if args.len() == 1 => Some(args[0].abs()),
        "if" if args.len() == 3 => {
            if args[0] != 0.0 {
                Some(args[1])
            } else {
                Some(args[2])
            }
        }
        _ => None,
    }
}

pub fn debug_dump_ast(ast: &NetlistAst) {
    println!(
        "netlist ast: title={:?} statements={} errors={}",
        ast.title,
        ast.statements.len(),
        ast.errors.len()
    );
}

pub fn debug_dump_elaborated(elab: &ElaboratedNetlist) {
    println!(
        "netlist elab: instances={} controls={} errors={}",
        elab.instances.len(),
        elab.control_count,
        elab.error_count
    );
}

fn build_subckt_map(subckts: &[SubcktDef]) -> std::collections::HashMap<String, SubcktDef> {
    let mut map = std::collections::HashMap::new();
    for def in subckts {
        map.insert(def.name.clone(), def.clone());
    }
    map
}

fn expand_subckt_instance_recursive(
    instance: &DeviceStmt,
    def: &SubcktDef,
    subckts: &std::collections::HashMap<String, SubcktDef>,
    local_params: &std::collections::HashMap<String, String>,
    global_params: &std::collections::HashMap<String, String>,
    errors: &mut Vec<ParseError>,
    models: &mut Vec<ControlStmt>,
) -> Vec<DeviceStmt> {
    let (body, nested_subckts, nested_errors) = extract_subckts(&def.body);
    errors.extend(nested_errors);
    let mut nested_map = build_subckt_map(&nested_subckts);
    for (name, def) in subckts {
        nested_map.entry(name.clone()).or_insert_with(|| def.clone());
    }

    // 构建端口映射：子电路端口 -> 实例节点
    let mut port_map = std::collections::HashMap::new();
    for (port, node) in def.ports.iter().zip(instance.nodes.iter()) {
        port_map.insert(port.to_ascii_lowercase(), node.clone());
    }

    let mut expanded = Vec::new();
    for stmt in body {
        match stmt {
            Stmt::Device(dev) => {
                let mut scoped = dev.clone();
                scoped.name = format!("{}.{}", instance.name, dev.name);
                scoped.nodes = dev
                    .nodes
                    .iter()
                    .map(|node| map_subckt_node(instance, &port_map, node))
                    .collect();
                if matches!(scoped.kind, DeviceKind::X) {
                    if let Some(subckt_name) = scoped.model.as_deref() {
                        if let Some(child_def) = nested_map.get(subckt_name) {
                            let body_params = collect_params_from_body(&child_def.body);
                            let child_params = build_local_param_table(
                                child_def,
                                &scoped,
                                &body_params,
                                local_params,
                                global_params,
                            );
                            let child_expanded = expand_subckt_instance_recursive(
                                &scoped,
                                child_def,
                                &nested_map,
                                &child_params,
                                global_params,
                                errors,
                                models,
                            );
                            expanded.extend(child_expanded);
                            continue;
                        }
                    }
                    errors.push(ParseError {
                        line: scoped.line,
                        message: format!("子电路未定义: {:?}", scoped.model),
                    });
                }

                let mut final_inst = scoped.clone();
                apply_params_to_device_scoped(global_params, local_params, &mut final_inst);
                expanded.push(final_inst);
            }
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Model) => {
                // Extract .model statements from subcircuits
                // Scope the model name to avoid conflicts
                let mut scoped_model = ctrl.clone();
                if let Some(ref name) = scoped_model.model_name {
                    scoped_model.model_name = Some(format!("{}.{}", instance.name, name));
                }
                models.push(scoped_model);
            }
            _ => {
                // Comments, .param (handled separately), and other control statements are ignored
            }
        }
    }

    expanded
}
