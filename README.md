# satmig

把 Paper Library 里的 live migration 工作（PipeLive / SpotServe / Llumnix / CONNEX /
SCOPE）移植到 LEO 卫星星座上，做成一个**配置驱动、单命令入口、确定性复跑**的仿真器，
并给出四个可行方案的量化验证。

承接 [LAB-47](https://github.com/) 的归约模型（单面、单流水线、二值可用性），把它删掉
的三样东西放回来：**时延**、**多流水线**、**异质轨道面**。

## 一句话结论

> 在 80–160 ms 的交互式 TPOT 预算下，把流水线摆成一串 ISL 邻居（地面 PP 的默认几何，
> 也是 LAB-47 的模型）**没有任何可行的切分数**——显存要求 P ≥ 10，而 1×P 沿轨弧的
> 往返跳预算只允许 P ≤ 3…9。换成 `w × ⌈P/w⌉` 的蛇形块后，TPOT 从 160.3 ms 降到
> 71.3 ms（−55.5%），可行窗口开到 P ∈ [10, 30]，最低可行 SLO 从 220 ms 降到 80 ms。
> 在此之上，截止期驱动的批量交接把停顿从 1.511% 压到 0.024%，能源空闲度 + 跨面重定位
> 在 8 条流水线争用 2 个轨道面时把聚合吞吐从 348.8 提到 770.3 tok/s（2.21×），
> 而近至日点直接选无蚀面可以把迁移次数降到 0。

完整表格与图见 `docs/PROPOSALS.md` 与 `results/report/report.md`。

## 跑起来

```sh
python3 -m unittest discover -s tests -t .      # 144 个测试
python3 -m satmig all --out results             # 六组实验，约 25 s
python3 -m satmig report --results results --out results/report
```

单个配置：

```sh
python3 -m satmig run   --config configs/base.yaml       --out results/base
python3 -m satmig sweep --config configs/exp4_slo.yaml   --out results/exp4_slo
```

只需要 Python 3.11+、`pyyaml`、`matplotlib`（仅出图用）。没有 scipy 依赖——KM 匹配是
自带的 O(n³) 实现，有对暴力枚举的一致性测试。

## 每次运行产出的四类文件

| 文件 | 内容 |
|---|---|
| `manifest.json` | 配置全量 + 指纹 + git rev + 派生常量 + **模型边界声明** |
| `slots.csv` | 逐 slot × 逐策略 × 逐流水线的 17 列原始记录 |
| `policy_metrics.csv` | 每策略一行的聚合指标 |
| `summary.json` | 头条数字 + 相对基线的比较 + 可行性判定 |

确定性：决策路径上没有任何随机数（光照是闭式的，策略是 `(state, t)` 的纯函数）。
`tests/test_e2e.py` 断言两次运行的三个文件**逐字节相同**，manifest 只差时间戳。

## 目录

```
satmig/
  orbit.py       闭式光照：周期、β_crit、蚀占比、本影中心、J2 漂移
  topology.py    (plane, slot) 环面格、沿轨/跨面跳时延、弧/列/蛇形块、往返跳
  perf.py        1F1B 吞吐、TPOT、PipeLive MaxBlocks、Llumnix 空闲度
  migration.py   SCOPE Eq.1-3、增量 KV 补丁、CONNEX cutover、ISL 公平共享
  matching.py    Kuhn-Munkres（SpotServe Device Mapper）
  policies.py    4 条基线 + 4 个提出的策略
  simulator.py   逐 slot 驱动、争用定价、可行性判定
  results.py     四类结果文件
  report.py      图 + Markdown/HTML 报告
configs/         base + exp1..exp6
docs/            MODEL.md（模型与公式出处）、PROPOSALS.md（四个方案）
tests/           144 个测试
```

## 八个策略

| 名字 | 来源 | 打破的假设 |
|---|---|---|
| `static` | LAB-47 baseline A | —— |
| `reactive_jit` | SpotServe (ASPLOS'24) | 抢占不可预报 ⇒ 宽限期短、无法预铺 |
| `llumnix_reactive` | Llumnix (2024) | 空闲度是**观测**量 |
| `conveyor` | LAB-47 baseline B | 每 `T/N` 必须平移一跳 |
| `eph` | 提出 | 截止期驱动 + 弧余量批量跳 + 错相 |
| `eph_compact` | 提出 | + 蛇形块（跨面跳 + 收窄回程） |
| `eph_freeness` | 提出 | + 能源空闲度 + 跨面重定位 + KM 映射 |
| `two_timescale` | 提出 | + 按面带的 β 排序，周级重规划 |

## 可信度分层

**已与本工作区独立推导对拍**（`tests/test_orbit.py` 里是断言）：轨道周期 95.5 min、
无蚀临界角 67.0°、β=0 蚀占比 0.372、单 shell 内 β 跨度到 `i+δ_s`、闭式 `p_full`、
KV-only 交接气泡 ~1.4%、每星每轨约 162 Wh 的蚀期计算能耗，以及 LAB-47 的
`P* = ⌊N(1−f)⌋` 断崖（P=13→14 时 −44.4%）。这些数字 LAB-47 / LAB-58 / LAB-67 是用
satellite.js + SGP4 算的，本仓库是用闭式算的，两条独立路径一致。

**照搬已发表工作的常数**：CONNEX 的 11.1 ms pair-local cutover 与约 50× 更慢的全局
重建；PipeLive 的 MaxBlocks、层堆叠粒度、增量 KV 补丁；SCOPE 的 Eq. 1–3；SpotServe 的
KM 设备映射与 30 s 宽限期；Llumnix 的空闲度公式。

**我们的假设，未标定**：归一化算力代理、蚀期降额 `c_e`（E3 专门扫）、按 42 ms 整模型
decode 反推的每层时间、每星 24 GB 可用显存、决定 pre-copy 收敛的 KV 脏写率。

**未建模**：姿态与帆板入射角（所以此处的"无功率时间"是**下界**）、电池 SoC、半影、
大气阻力、J2 短周期项、prefill/decode 混合、请求到达过程、slot 内排队。

## 已知的局限

- 交接在其起始 slot 内定价（10 s slot ≫ 0.4 s 交接），slot 内排队未解析。
- 面间链路只在**同 slot** 相连；真实星座的跨缝（seam）与高纬度链路重构未建模。
- 只做星上放置，不含星地链路与用户侧——那是 SCOPE 的分工。
- `eph` 在同面有邻居时会被卡住（E2 里 K≥2 输给 conveyor）。这是有意留着的消融项，
  说明面内批量化必须配跨面重定位才成立。
