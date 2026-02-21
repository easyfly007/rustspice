# RustSpice 用户手册

本文档面向使用 RustSpice 的用户，介绍基本使用流程、命令行模式以及如何读取仿真输出。当前版本以 DC 与 TRAN 的基本流程为目标，随着功能完善会持续更新。

## 1. 快速开始

### 1.1 准备 netlist

RustSpice 使用 SPICE 风格网表文件作为输入。示例:

```
* Basic DC example
V1 in 0 DC 1
R1 in out 1k
R2 out 0 2k
.op
.end
```

保存为 `example.cir`。

### 1.2 运行仿真

当前的二进制入口为 `sim-cli`:

```
sim-cli example.cir
```

运行成功会输出节点电压（OP 结果）。

如需输出 PSF 文本:

```
sim-cli example.cir --psf /tmp/example.psf
```

指定分析类型（OP / DC）:

```
sim-cli example.cir --analysis op
```

```
sim-cli example.cir --analysis dc --dc-source V1 --dc-start 0 --dc-stop 1 --dc-step 0.1
```

### 1.3 启动 API 服务

```
cargo run -p sim-api -- --addr 127.0.0.1:3000
```

使用 netlist 字符串触发 OP:

```
curl -X POST http://127.0.0.1:3000/v1/run/op \
  -H "Content-Type: application/json" \
  -d "{\"netlist\":\"V1 in 0 DC 1\\nR1 in out 1k\\nR2 out 0 2k\\n.op\\n.end\\n\"}"
```

使用文件路径触发 OP（相对路径相对于服务启动目录）:

```
curl -X POST http://127.0.0.1:3000/v1/run/op \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"tests/fixtures/netlists/basic_dc.cir\"}"
```

触发 DC 扫描:

```
curl -X POST http://127.0.0.1:3000/v1/run/dc \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"tests/fixtures/netlists/basic_dc.cir\",\"source\":\"V1\",\"start\":0,\"stop\":1,\"step\":0.1}"
```

触发 TRAN 分析:

```
curl -X POST http://127.0.0.1:3000/v1/run/tran \
  -H "Content-Type: application/json" \
  -d "{\"netlist\":\"V1 in 0 DC 1\\nR1 in 0 1k\\n.tran 1e-6 1e-5\\n.end\\n\"}"
```

查询运行结果:

```
curl http://127.0.0.1:3000/v1/runs
curl http://127.0.0.1:3000/v1/runs/0
```

导出 PSF 文本:

```
curl -X POST http://127.0.0.1:3000/v1/runs/0/export \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"/tmp/run0.psf\"}"
```

查询电路结构:

```
curl http://127.0.0.1:3000/v1/summary
curl http://127.0.0.1:3000/v1/nodes
```

---

## 2. 命令行模式 (Binary Mode)

### 2.1 基本语法

```
sim-cli <netlist>
```

- `<netlist>`: SPICE 网表文件路径  
- 网表必须包含 `.end`

### 2.3 当前解析能力说明

当前 parser 已具备以下能力:

- 行级解析与续行拼接
- 识别注释行与空行
- 识别控制语句与器件语句
- 抽取器件节点与值/模型 (简化规则)
- 参数按 `key=value` 形式抽取
- 识别 `.param` / `.model` / `.subckt` (基础字段)
- 部分器件节点与字段的基础校验
- MOS 支持 3 节点（隐式 bulk=0）
- 支持 `.include` 递归读取 (基础版)
- 支持子电路基础展开与参数映射 (简单覆盖规则)
- 支持基础参数表达式求值 (加减乘除与括号，支持单位后缀)
- 受控源 E/G/F/H 支持 POLY 语法解析与系数校验
- POLY 控制节点/控制源数量校验
- 支持嵌套子电路基础展开
- 支持参数表达式函数: max/min/abs/if，并支持一元负号与幂运算 (^)
- 电压/电流源支持 DC 关键字取值
- 电压/电流源支持波形关键字（PULSE/SIN/AC 等）
- 参数支持逗号分隔，model 括号参数解析
- 波形关键字缺参校验
- 支持子电路内部 `.param` 作用域
- .model 参数已合并到实例参数（模型参数为默认值，实例参数可覆盖）

注意:

- 节点数量与字段语义尚未严格校验
- 参数表达式与更复杂的作用域规则将在后续阶段完善

### 2.2 批量运行 (spice-datasets)

项目支持批量运行 `spice-datasets` 中的网表，用于 smoke test:

```
python tests/run_spice_datasets.py
```

执行后会输出通过率，例如:

```
total=50 passed=50 failed=0 passrate=100.00%
```

---

## 3. DC Sweep 分析

### 3.1 基本语法

在网表中使用 `.dc` 命令定义扫描:

```spice
.dc <source> <start> <stop> <step>
```

参数说明:
- `<source>`: 要扫描的电压源或电流源名称 (如 V1, I1)
- `<start>`: 起始值
- `<stop>`: 结束值
- `<step>`: 步进值

### 3.2 示例

```spice
* DC Sweep - 电阻分压器
V1 in 0 DC 0
R1 in out 1k
R2 out 0 2k
.dc V1 0 5 0.5
.end
```

此示例将 V1 从 0V 扫描到 5V，步进 0.5V (共 11 个点)。

### 3.3 支持特性

- **正向/反向扫描**: start 可以大于或小于 stop
- **单点扫描**: start == stop 时只计算一个点
- **解的连续性**: 使用前一点的解作为下一点的初始值，提高收敛性
- **负电压扫描**: 支持负电压范围

### 3.4 命令行使用

```bash
# 使用网表中的 .dc 命令
sim-cli example.cir

# 通过命令行参数指定扫描
sim-cli example.cir --analysis dc --dc-source V1 --dc-start 0 --dc-stop 5 --dc-step 0.5
```

### 3.5 API 使用

```bash
curl -X POST http://127.0.0.1:3000/v1/run/dc \
  -H "Content-Type: application/json" \
  -d '{"path":"example.cir","source":"V1","start":0,"stop":5,"step":0.5}'
```

### 3.6 结果格式

DC sweep 的结果包含:
- `sweep_var`: 扫描变量名
- `sweep_values`: 各扫描点的值
- `sweep_solutions`: 各扫描点的完整解向量

---

## 4. 受控源与 POLY 语法

RustSpice 支持四种受控源（Controlled Sources），可以用于构建运算放大器模型、电流镜、依赖源等电路结构。

### 4.1 基本受控源

#### VCVS (E) - 电压控制电压源

输出电压与控制电压成正比：`Vout = gain × Vin`

```spice
* 语法: E<name> <out+> <out-> <in+> <in-> <gain>
E1 out 0 in 0 2.0       ; Vout = 2 × V(in)
```

#### VCCS (G) - 电压控制电流源

输出电流与控制电压成正比：`Iout = gm × Vin`

```spice
* 语法: G<name> <out+> <out-> <in+> <in-> <transconductance>
G1 out 0 in 0 0.001     ; Iout = 0.001 × V(in)
```

#### CCCS (F) - 电流控制电流源

输出电流与控制电流成正比：`Iout = gain × Iin`

```spice
* 语法: F<name> <out+> <out-> <Vcontrol> <gain>
* Vcontrol 是用于检测控制电流的电压源名称
F1 out 0 Vsense 3.0     ; Iout = 3 × I(Vsense)
```

#### CCVS (H) - 电流控制电压源

输出电压与控制电流成正比：`Vout = transresistance × Iin`

```spice
* 语法: H<name> <out+> <out-> <Vcontrol> <transresistance>
H1 out 0 Vsense 1k      ; Vout = 1000 × I(Vsense)
```

### 4.2 POLY 语法详解

POLY（Polynomial，多项式）是受控源的高级语法，允许定义**非线性多项式关系**或**多输入依赖**。

#### 基本格式

```spice
* POLY(n) 表示 n 组控制输入
* 对于 E/G (电压控制): 每组控制输入需要 2 个节点 (正负端)
* 对于 F/H (电流控制): 每组控制输入需要 1 个电压源名称

E<name> <out+> <out-> POLY(n) <ctrl1+> <ctrl1-> ... <ctrln+> <ctrln-> <c0> <c1> <c2> ...
F<name> <out+> <out-> POLY(n) <Vctrl1> <Vctrl2> ... <Vctrln> <c0> <c1> <c2> ...
```

#### 单输入多项式 POLY(1)

当 n=1 时，输出是单个控制变量的多项式函数：

```
Vout = c0 + c1×x + c2×x² + c3×x³ + ...
```

其中 x 是控制电压或控制电流。

**示例：**

```spice
* 二次多项式: Vout = 0.5 + 2.0×V(in) + 0.1×V(in)²
E1 out 0 POLY(1) in 0 0.5 2.0 0.1

* 等效于线性源 (只有 c0 和 c1):
E2 out 0 POLY(1) in 0 0.0 2.0   ; Vout = 2.0 × V(in)
```

#### 多输入多项式 POLY(n)

当 n>1 时，系数按以下顺序对应多项式项：

对于 POLY(2)，变量为 x1 和 x2：
```
Vout = c0 + c1×x1 + c2×x2 + c3×x1×x2 + c4×x1² + c5×x2² + ...
```

**示例：**

```spice
* 双输入 POLY(2): 控制节点 a-0 和 b-0
* Vout = 1.0 + 2.0×V(a) + 3.0×V(b) + 4.0×V(a)×V(b)
E1 out 0 POLY(2) a 0 b 0 1.0 2.0 3.0 4.0

* 电流控制版本
F1 out 0 POLY(2) V1 V2 1.0 2.0 3.0
```

#### 系数数量要求

对于 POLY(n)，最少需要 n+1 个系数（c0 到 cn）：

| POLY(n) | 控制节点/源 | 最少系数 |
|---------|-------------|----------|
| POLY(1) | 2 节点 或 1 源 | 2 (c0, c1) |
| POLY(2) | 4 节点 或 2 源 | 3 (c0, c1, c2) |
| POLY(3) | 6 节点 或 3 源 | 4 (c0, c1, c2, c3) |

### 4.3 POLY 典型应用

#### 乘法器 (Multiplier)

```spice
* 模拟乘法器: Vout = V(a) × V(b)
* 使用 POLY(2)，系数 c3 对应 x1×x2 项
E_mult out 0 POLY(2) a 0 b 0 0 0 0 1.0
```

#### 平方器 (Squarer)

```spice
* 平方运算: Vout = V(in)²
* 使用 POLY(1)，系数 c2 对应 x² 项
E_sq out 0 POLY(1) in 0 0 0 1.0
```

#### 加法器 (Summer)

```spice
* 加权求和: Vout = 2×V(a) + 3×V(b)
E_sum out 0 POLY(2) a 0 b 0 0 2.0 3.0
```

### 4.4 当前实现状态

| 功能 | 状态 |
|------|------|
| POLY 语法**解析** | ✅ 已实现 |
| POLY 系数**校验** | ✅ 已实现 |
| 控制节点/源数量**验证** | ✅ 已实现 |
| 线性受控源**仿真** (无 POLY 或 POLY(1) 仅 c1) | ✅ 已实现 |
| POLY(1) 多项式**仿真** (单输入多项式) | ✅ 已实现 |
| POLY(2) 多项式**仿真** (双输入多项式) | ✅ 已实现 |
| POLY(n) 多项式**仿真** (n≥3，线性项) | ✅ 已实现 |
| AC 分析 POLY 小信号模型 | ✅ 已实现 |

**技术实现：**

- 多项式受控源使用 **Newton-Raphson 迭代** 求解非线性方程
- 在每次迭代中，多项式在当前工作点线性化
- 对于 POLY(1)：支持任意阶多项式 (c0 + c1*x + c2*x² + ...)
- 对于 POLY(2)：支持到交叉项 (c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1² + c5*x2²)
- AC 分析使用 DC 工作点的小信号导数

---

## 5. 输出格式

RustSpice 支持多种输出格式，可通过 `-f` 或 `--format` 选项指定。

### 5.1 支持的格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PSF | `.psf` | Cadence PSF 文本格式（默认） |
| Raw | `.raw` | ngspice/LTspice 兼容格式 |
| JSON | `.json` | 结构化 JSON 格式 |
| CSV | `.csv` | 逗号分隔值格式 |

### 5.2 命令行使用

```bash
# PSF 格式（默认）
sim-cli circuit.cir -o output.psf

# ngspice Raw 格式
sim-cli circuit.cir -o output.raw -f raw

# JSON 格式
sim-cli circuit.cir -o output.json -f json

# CSV 格式
sim-cli circuit.cir -o output.csv -f csv
```

### 5.3 JSON 格式详解

JSON 格式提供结构化数据输出，包含元信息和仿真数据。

#### OP 分析输出

```json
{
  "format": "rustspice-json",
  "version": "0.1.0",
  "analysis": "Op",
  "variables": [
    {"name": "vdd", "type": "voltage", "value": 5.0},
    {"name": "out", "type": "voltage", "value": 2.5}
  ]
}
```

#### DC Sweep 输出

```json
{
  "format": "rustspice-json",
  "version": "0.1.0",
  "analysis": "Dc",
  "sweep_source": "V1",
  "points": 6,
  "variables": [
    {"name": "V1", "type": "sweep"},
    {"name": "in", "type": "voltage"},
    {"name": "out", "type": "voltage"}
  ],
  "data": [
    [0.0, 5.0, 3.333333],
    [1.0, 5.0, 3.333333]
  ]
}
```

#### TRAN 分析输出

```json
{
  "format": "rustspice-json",
  "version": "0.1.0",
  "analysis": "Tran",
  "points": 100,
  "variables": [
    {"name": "time", "type": "time"},
    {"name": "in", "type": "voltage"},
    {"name": "out", "type": "voltage"}
  ],
  "data": [
    [0.0, 1.0, 0.5],
    [1e-6, 1.0, 0.632]
  ]
}
```

#### AC 分析输出

```json
{
  "format": "rustspice-json",
  "version": "0.1.0",
  "analysis": "Ac",
  "points": 10,
  "variables": [
    {"name": "frequency", "type": "frequency"},
    {"name": "out", "type": "complex"}
  ],
  "data": [
    [1.0, {"magnitude_dB": 0.0, "phase_deg": 0.0}],
    [10.0, {"magnitude_dB": -0.04, "phase_deg": -5.7}]
  ]
}
```

### 5.4 CSV 格式详解

CSV 格式提供简单的逗号分隔数据，便于导入 Excel、Python pandas 等工具。

#### OP 分析输出

```csv
node,type,value
vdd,voltage,5.000000e0
out,voltage,2.500000e0
```

#### DC Sweep 输出

```csv
V1,V(in),V(out)
0.000000e0,5.000000e0,3.333333e0
1.000000e0,5.000000e0,3.333333e0
```

#### TRAN 分析输出

```csv
time,V(in),V(out)
0.000000e0,1.000000e0,5.000000e-1
1.000000e-6,1.000000e0,6.321206e-1
```

#### AC 分析输出

```csv
frequency,out_dB,out_deg
1.000000e0,0.000000e0,0.000000e0
1.000000e1,-4.000000e-2,-5.700000e0
```

### 5.5 结果解读建议

- **OP**: 查看静态工作点电压与电流
- **DC Sweep**: 查看参数扫描曲线，验证电路传输特性
- **TRAN**: 查看时域波形
- **AC**: 查看频率响应（幅度 dB 和相位度）

当前模型参数支持（基础版）:
- 二极管: `IS`、`N`/`NJ`
- MOS: `VTH`/`VTO`、`BETA`/`KP`、`LAMBDA`

---

## 6. 仿真选项 (.option)

使用 `.option` 指令可以控制仿真器的各项参数。在网表中添加一行或多行 `.option` 即可:

```spice
.option <key>=<value>
```

也可以在一行中设置多个选项:

```spice
.option abstol=1e-15 reltol=1e-4
```

### 6.1 选项列表

| 选项 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `abstol` | float | 1e-12 | (0, 1e-3) | 绝对电流容差 |
| `reltol` | float | 1e-3 | (0, 1.0) | 相对容差 |
| `vntol` | float | 1e-6 | (0, 1.0) | 电压节点容差 |
| `gmin` | float | 1e-12 | (0, 1e-3) | 最小电导 |
| `itl1` | int | 100 | [1, 10000] | DC 迭代上限 |
| `itl4` | int | 50 | [1, 10000] | TRAN 迭代上限 |
| `temp` | float | 27.0 | (-273.15, 1000) | 仿真温度 (°C) |
| `tnom` | float | 27.0 | (-273.15, 1000) | 标称温度 (°C) |
| `solver` | string | auto | 见下表 | 线性求解器选择 |

### 6.2 求解器选项 (solver)

`solver` 选项控制仿真引擎使用的线性求解器。默认值为 `auto`，自动根据矩阵大小选择最合适的求解器。

```spice
.option solver=auto        * 自动选择（默认）
.option solver=nativeklu   * 使用 Native KLU 求解器
.option solver=dense       * 使用稠密 LU 求解器
```

可用的求解器:

| 值 | 求解器 | 说明 |
|----|--------|------|
| `auto` | 自动选择 | 根据矩阵大小自动选择最佳求解器（默认） |
| `dense` | Dense LU | O(n³)，适合小规模矩阵 (n < 100) |
| `sparse` / `sparselu` | Sparse LU | O(nnz·fill)，纯 Rust 实现 |
| `sparselubtf` | Sparse LU + BTF | 带 BTF 分解，适合块结构矩阵 |
| `bbd` | BBD | 边界块对角分解 + Schur 补 |
| `faer` | Faer | 纯 Rust 稀疏求解器（需 `faer-solver` feature） |
| `klu` | KLU (C) | SuiteSparse KLU（需 `klu` feature 及 C 库） |
| `nativeklu` | Native KLU | 纯 Rust KLU 实现，支持 BTF/AMD/并行重分解 |

**自动选择规则** (`auto`):

- n ≤ 50: Dense（开销低）
- 50 < n ≤ 500: 优先 KLU → Faer → SparseLU
- n > 500: 优先 KLU → Faer → SparseLU-BTF

### 6.3 示例

```spice
* 使用 Native KLU 求解器，提高迭代上限
V1 in 0 DC 5
R1 in out 1k
R2 out 0 2k
.option solver=nativeklu itl1=200
.op
.end
```

---

## 7. 常见问题

### 7.1 提示缺少 .end

确保网表末尾有 `.end` 行。

### 7.2 无输出结果

当前版本已支持基础求解输出。若无输出，通常是求解失败或网表语法不完整，请先检查错误提示。

### 7.3 DC Sweep 收敛失败

如果在某个扫描点收敛失败:
- 检查电路是否有效 (是否有浮空节点)
- 尝试减小步进值
- 检查初始值是否在有效范围内

---

## 8. 版本说明

手册会与项目阶段同步更新。后续将补充:

- 仿真结果字段说明
- 交互式 API 使用方法
- CLI 与 AI 代理的使用示例
- KLU 求解器的使用与依赖说明
- 仿真引擎核心结构说明 (Circuit/MNA/Solver)
- 仿真引擎骨架的使用与调试说明

当前进展:

- 已加入 DC 基础 stamp（R/I/V/D/MOS）
- MNA 骨架与单元测试已覆盖
- KLU 接口为 stub，需链接 SuiteSparse 后启用
- Newton 迭代与收敛控制骨架已加入
- gmin/source stepping 基础接入
- 二极管/MOS 非线性线性化（简化模型）
- TRAN 骨架入口（时间步循环）
- TRAN 电容/电感等效 stamp
- TRAN 自适应步长与误差估计骨架
- TRAN 非线性器件 Newton 迭代
- ResultStore 接入与仿真结果输出
- PSF 文本输出（基础格式）
- TRAN 收敛失败时回退 gmin/source stepping
- TRAN 加权误差估计

## 9. KLU 求解器依赖说明

RustSpice 的线性求解器规划为 SuiteSparse 的 KLU。当前阶段仅完成规划与接口设计，正式启用时需要:

- 安装 SuiteSparse（包含 KLU）
- 在 Windows 环境准备预编译库或本地编译产物

### 9.1 安装流程（草案）

#### Windows（推荐）

1) 安装 Visual Studio Build Tools（包含 C/C++ 工具链）
2) 获取 SuiteSparse 预编译包或自行编译
3) 配置环境变量或在构建系统中指定库路径

建议路径约定:

- 头文件: `C:\libs\suitesparse\include`
- 库文件: `C:\libs\suitesparse\lib`

#### Linux

1) 使用系统包管理器安装 SuiteSparse  
   - Ubuntu 示例: `sudo apt install libsuitesparse-dev`
2) 确认 `klu.h` 与库文件可被构建系统找到

#### macOS

1) 使用 Homebrew 安装  
   - `brew install suite-sparse`
2) 确认库路径可被 Rust 构建系统访问

### 9.2 构建系统配置（草案）

建议采用 `build.rs` + 环境变量的方式自动发现 KLU。

环境变量约定（示例）:

- `SUITESPARSE_DIR`：SuiteSparse 根目录  
- `KLU_INCLUDE_DIR`：头文件目录  
- `KLU_LIB_DIR`：库文件目录  

构建系统行为（建议）:

1) 优先读取 `KLU_INCLUDE_DIR` / `KLU_LIB_DIR`
2) 若未设置，则尝试 `SUITESPARSE_DIR/include` 与 `SUITESPARSE_DIR/lib`
3) 最后尝试系统默认路径

---

## 10. BSIM 模型文档

完整说明请参考：`docs/bsim_model.md`。

示例（Windows PowerShell）:

```
$env:SUITESPARSE_DIR="C:\libs\suitesparse"
```

示例（Linux/macOS）:

```
export SUITESPARSE_DIR=/usr/local
```