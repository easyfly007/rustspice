# BSIM 模型说明（RustSpice 版本）

## 1. BSIM 是什么

BSIM（Berkeley Short-channel IGFET Model）是由 UC Berkeley BSIM 研究组维护的 MOSFET 紧凑模型系列。
BSIM3/BSIM4 覆盖短沟道效应、迁移率退化、速度饱和、DIBL、寄生电阻电容等多种物理效应，是工业界主流 SPICE MOS 模型。

参考：https://bsim.berkeley.edu/

> **更新（v1.0）**：RustSpice 现已实现 **BSIM3v3 (LEVEL=49) 核心 DC 模型**，支持完整的阈值电压、迁移率、输出电导计算。

---

## 2. 支持的模型级别

| Level | 模型名称 | 状态 | 说明 |
|-------|----------|------|------|
| 1 | Shichman-Hodges | ✅ 支持 | 简化教学级模型 |
| 49 | BSIM3v3 | ✅ 支持 | 工业标准 DC 模型 |
| 54 | BSIM4 | ❌ 未实现 | 计划中 |

---

## 3. BSIM3 核心物理模型

### 3.1 阈值电压 (Vth)

```
Vth = VTH0 + ΔVth_body + ΔVth_SCE + ΔVth_DIBL + ΔVth_temp

其中：
  VTH0        = 零偏阈值电压
  ΔVth_body   = K1·(√(φ-Vbs) - √φ) + K2·Vbs      [体效应]
  ΔVth_SCE    = -DVT0·exp(-DVT1·Leff/2lt)·Vt     [短沟道效应]
  ΔVth_DIBL   = -ETA0·Vds                         [漏致势垒降低]
  ΔVth_temp   = KT1·(T/Tnom - 1)                  [温度效应]
```

### 3.2 载流子迁移率

```
μeff = μ0 · Ftemp / (1 + UA·Eeff + UB·Eeff² + UC·Vbs·Eeff)

其中：
  μ0    = 低场迁移率 [cm²/V/s]
  Ftemp = (T/Tnom)^UTE                    [温度因子]
  Eeff  = (Vgs - Vth + 2Vt) / (6·tox)     [有效垂直电场]
```

### 3.3 饱和电压

```
Vdsat = Vgst · Esat·Leff / (Vgst + Esat·Leff)

其中：
  Vgst = Vgs - Vth                        [栅过驱动]
  Esat = 2·VSAT / μeff                    [饱和电场]
```

### 3.4 工作区域

| 区域 | 条件 | 电流方程 |
|------|------|----------|
| 截止区 | Vgs < Vth | Ids ≈ 0（亚阈值漏电） |
| 线性区 | Vds < Vdsat | Ids = β[(Vgs-Vth)Vds - Vds²/2] |
| 饱和区 | Vds ≥ Vdsat | Ids = β·Vdsat²/2 · CLM_factor |

### 3.5 沟道长度调制 (CLM)

```
CLM_factor = 1 / (1 - ΔL/L)

其中：
  ΔL/L = PCLM · (Vds - Vdsat) / (Esat · Leff)
```

---

## 4. 支持的参数

### 4.1 阈值电压参数

| 参数 | NMOS 默认 | PMOS 默认 | 单位 | 说明 |
|------|-----------|-----------|------|------|
| VTH0 | 0.7 | -0.7 | V | 零偏阈值电压 |
| K1 | 0.5 | 0.5 | V^0.5 | 一阶体效应系数 |
| K2 | 0.0 | 0.0 | - | 二阶体效应系数 |
| DVT0 | 2.2 | 2.2 | - | 短沟道效应系数 |
| DVT1 | 0.53 | 0.53 | - | 短沟道效应指数 |
| DVT2 | -0.032 | -0.032 | 1/V | 体偏压 SCE 系数 |
| ETA0 | 0.08 | 0.08 | - | DIBL 系数 |
| DSUB | 0.56 | 0.56 | - | DIBL 指数系数 |
| NLX | 1.74e-7 | 1.74e-7 | m | 窄沟道效应参数 |
| NFACTOR | 1.0 | 1.0 | - | 亚阈值摆幅因子 |

### 4.2 迁移率参数

| 参数 | NMOS 默认 | PMOS 默认 | 单位 | 说明 |
|------|-----------|-----------|------|------|
| U0 | 500 | 150 | cm²/V/s | 低场迁移率 |
| UA | 2.25e-9 | 2.25e-9 | m/V | 一阶迁移率退化 |
| UB | 5.87e-19 | 5.87e-19 | (m/V)² | 二阶迁移率退化 |
| UC | -4.65e-11 | -4.65e-11 | m/V² | 体偏压迁移率退化 |
| VSAT | 1.5e5 | 1.5e5 | m/s | 饱和速度 |

### 4.3 输出电导参数

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| PCLM | 1.3 | - | 沟道长度调制系数 |
| PDIBLC1 | 0.39 | - | DIBL 输出电阻系数 1 |
| PDIBLC2 | 0.0086 | - | DIBL 输出电阻系数 2 |
| DROUT | 0.56 | - | DIBL 长度依赖 |

### 4.4 几何参数

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| TOX | 1.5e-8 | m | 栅氧化层厚度 |
| LINT | 0.0 | m | 沟道长度偏移 (Leff = L - 2·LINT) |
| WINT | 0.0 | m | 沟道宽度偏移 (Weff = W - 2·WINT) |

### 4.5 寄生电阻参数

| 参数 | 默认值 | 单位 | 说明 |
|------|--------|------|------|
| RDSW | 0.0 | Ω·μm | 源漏串联电阻（每单位宽度） |

### 4.6 温度参数

| 参数 | NMOS 默认 | PMOS 默认 | 单位 | 说明 |
|------|-----------|-----------|------|------|
| TNOM | 300.15 | 300.15 | K | 标称温度 (27°C) |
| UTE | -1.5 | -1.0 | - | 迁移率温度指数 |
| KT1 | -0.11 | -0.08 | V | Vth 温度系数 |
| KT2 | 0.022 | 0.022 | V | Vth 温度系数（体偏压） |

---

## 5. 网表语法

### 5.1 基本 MOSFET 实例

```spice
* NMOS 晶体管
M1 drain gate source bulk NMOS W=1u L=100n

* PMOS 晶体管
M2 drain gate source bulk PMOS W=2u L=100n

* 带参数的实例
M3 drain gate source bulk NMOS W=1u L=100n VTH0=0.4 U0=400
```

### 5.2 模型定义

```spice
* BSIM3 模型定义
.model NMOS NMOS (LEVEL=49 VTH0=0.4 U0=400 TOX=2e-9 K1=0.5)
.model PMOS PMOS (LEVEL=49 VTH0=-0.4 U0=150 TOX=2e-9 K1=0.5)

* Level 1 简化模型（向后兼容）
.model NMOS1 NMOS (LEVEL=1 VTO=0.7 KP=1e-3 LAMBDA=0.02)
```

### 5.3 参数合并规则

1. `.model` 参数作为默认值
2. 器件实例参数覆盖 `.model` 参数

例如：
```spice
.model NMOS NMOS LEVEL=49 VTH0=0.7 U0=500
M1 d g s b NMOS VTH0=0.5 W=1u L=100n
```
最终 `M1.VTH0 = 0.5`，`M1.U0 = 500`。

---

## 6. 小信号模型 (MNA Stamp)

### 6.1 线性化方程

MOSFET 电流线性化为：
```
i_ds = gm·(vgs - VGS) + gds·(vds - VDS) + gmbs·(vbs - VBS) + IDS
     = gm·vgs + gds·vds + gmbs·vbs + ieq

其中：
  ieq = IDS - gm·VGS - gds·VDS - gmbs·VBS
```

### 6.2 MNA 矩阵 Stamp

```
         D    G    S    B    RHS
    ┌                              ┐
D   │  gds  gm  -gds-gm  gmbs  -ieq│
G   │   0   0    0       0     0  │
S   │ -gds -gm  gds+gm  -gmbs  ieq │
B   │   0   0    0       0     0  │
    └                              ┘
```

### 6.3 输出变量

| 变量 | 说明 | 单位 |
|------|------|------|
| ids | 漏源电流 | A |
| gm | 跨导 (∂Ids/∂Vgs) | S |
| gds | 输出电导 (∂Ids/∂Vds) | S |
| gmbs | 体跨导 (∂Ids/∂Vbs) | S |
| vth_eff | 有效阈值电压 | V |

---

## 7. 代码实现位置

### 7.1 模块结构

```
crates/sim-devices/src/bsim/
├── mod.rs          # 模块导出、参数构建、模型路由
├── params.rs       # BsimParams 结构（50+ 参数）
├── types.rs        # MosType, MosRegion, BsimOutput
├── threshold.rs    # Vth 计算（体效应、SCE、DIBL、温度）
├── mobility.rs     # μeff 计算（场退化、温度）
├── channel.rs      # Vdsat、CLM、输出电导
├── evaluate.rs     # DC 评估主函数
└── README.md       # 详细英文文档
```

### 7.2 关键函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `evaluate_bsim_dc()` | evaluate.rs | BSIM3 DC 评估主函数 |
| `evaluate_level1_dc()` | evaluate.rs | Level 1 兼容函数 |
| `evaluate_mos()` | mod.rs | 模型级别路由 |
| `build_bsim_params()` | mod.rs | 从 HashMap 构建参数 |
| `calculate_vth()` | threshold.rs | 阈值电压计算 |
| `calculate_mobility()` | mobility.rs | 迁移率计算 |
| `calculate_vdsat()` | channel.rs | 饱和电压计算 |
| `calculate_clm_factor()` | channel.rs | CLM 因子计算 |

### 7.3 Stamp 集成

位置：`crates/sim-core/src/stamp.rs` 的 `stamp_mos()` 函数

```rust
// 调用 BSIM 评估器
let output = sim_devices::bsim::evaluate_mos(
    &params, w, l, vd, vg, vs, vb, temp
);

// Stamp gds, gm, gmbs 到 MNA 矩阵
ctx.add(drain, drain, gds);
ctx.add(drain, gate, gm);
ctx.add(drain, bulk, gmbs);
// ...
```

---

## 8. 测试

### 8.1 运行测试

```bash
# 运行所有 BSIM 单元测试
cargo test -p sim-devices

# 运行集成测试
cargo test -p sim-core --test bsim_integration

# 运行特定测试
cargo test -p sim-devices bsim::evaluate::tests::test_nmos_saturation
```

### 8.2 测试覆盖

| 模块 | 测试数 | 覆盖内容 |
|------|--------|----------|
| threshold.rs | 4 | 体效应、DIBL、SCE、dVth/dVbs |
| mobility.rs | 3 | 基础迁移率、场退化、温度效应 |
| channel.rs | 4 | Vdsat、CLM 因子、线性/饱和区 |
| evaluate.rs | 7 | 所有区域、NMOS/PMOS、宽度缩放 |
| mod.rs | 5 | 参数解析、级别路由 |

---

## 9. 示例电路

### 9.1 CMOS 反相器

```spice
* CMOS Inverter
VDD vdd 0 DC 1.8
VIN in 0 DC 0.9

MP vdd in out vdd PMOS W=2u L=100n
MN out in 0 0 NMOS W=1u L=100n

.model NMOS NMOS (LEVEL=49 VTH0=0.4 U0=400)
.model PMOS PMOS (LEVEL=49 VTH0=-0.4 U0=150)

.op
.end
```

### 9.2 NMOS 电流镜

```spice
* NMOS Current Mirror
VDD vdd 0 DC 3.3
IREF vdd drain1 DC 100u

M1 drain1 drain1 0 0 NMOS W=10u L=1u
M2 drain2 drain1 0 0 NMOS W=10u L=1u
RL vdd drain2 10k

.model NMOS NMOS (LEVEL=49 VTH0=0.7 U0=500)

.op
.end
```

---

## 10. 限制与未来计划

### 10.1 当前限制

1. **仅 DC 分析**：尚未实现电容模型用于 AC/瞬态分析
2. **无噪声模型**：闪烁噪声参数已定义但未使用
3. **简化 Rds**：串联电阻效果为近似处理
4. **无 Binning**：不支持宽长插值
5. **无工艺变化**：不支持 Monte Carlo 参数

### 10.2 计划增强

1. **BSIM4 支持** (Level 54)
2. **电容模型**：Cgs, Cgd, Cgb
3. **AC 分析**：小信号模型
4. **噪声模型**：热噪声、闪烁噪声

---

## 11. 参考文献

1. **BSIM3v3 Manual**, UC Berkeley Device Group
2. **Y. Cheng, C. Hu**, "MOSFET Modeling & BSIM3 User's Guide"
3. **W. Liu**, "MOSFET Models for SPICE Simulation"
4. **BSIM Group Website**: http://bsim.berkeley.edu/

---

## 12. 更新日志

### v1.0.0 (当前版本)

- 实现 BSIM3v3 DC 模型 (Level 49)
- 支持 NMOS 和 PMOS 器件
- 完整阈值电压模型（体效应、SCE、DIBL）
- 迁移率退化与温度效应
- 沟道长度调制
- MNA stamping 集成
- 全面单元测试

### v0.x (历史版本)

- 教学级简化 MOS 模型
- 仅支持 VTH/KP/LAMBDA 参数
