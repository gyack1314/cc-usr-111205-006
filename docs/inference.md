# 显式持续时间推断：方法、输出与结果边界

本文件说明 `response_data.hsmm`（命令行 `infer`／`score`）的推断方法、输出字段和四类边界情形下的结果行为。观测与参数的冻结口径见[资料约定](data-contract.md)，此处不重复定义，只说明推断层的选择。

## 方法

隐状态以整段持续，段内不变；段间按冻结转移矩阵（对角为 0）转移；首时隙视为新段起点，首段用初始分布和普通持续分布 P(D=d)，不是平稳过程随机抽到一段中间。观测窗末段是尚未结束的右删失段，持续因子为 P(D>=已见长度)。

推断用显式持续时间的前向／后向递推（全部在 log 域）：

- `alpha[t][j]`：段 j 恰好结束于 t 的 log 联合概率（仅内部边界）；
- `beta[t][i]`：段 i 恰好结束于 t 时其后全部资料的 log 条件概率，其后第一段可以是精确结束段，也可以是删失末段；
- 资料似然：所有"末段状态 j × 已见长度 L"的删失段项之和；
- 状态后验 gamma：每个时隙上对覆盖它的所有段加权求和，按似然归一，每时隙和为 1；
- 边界转移后验 xi[t][i][j]：段 i 恰好结束于 t 且下一段为 j 的后验概率；全部边界、全部配对的和是预期段间转移数，不为 1；
- 最优分段路径：max-plus 动态规划（Viterbi 对应物），每个结点携带决胜签名。

短序列（合法分段数不超出 `--enumerate-cap`，默认 200000）自动做全枚举核对：似然、状态后验、边界转移后验、最优路径都与动态规划结果对照，差异列在输出的 `checks.enumeration` 下（本资料库三份文件均在 1e-13 以内一致）。超过上限时退化为仅动态规划，`best_path.selection` 为 `dp`，并在 `diagnostics` 中说明。

## 输出字段（infer）

- `log_likelihood`：资料似然的 log；末段一律按删失处理。
- `state_posterior`：每时隙各状态后验概率、`map_state`（边际后验最大标注，并列取 state_order 名次小者）、`path_state`（最优整路径经过的状态）及二者是否不一致。两种口径不同，可以不同，输出分别给出。
- `transition_posterior`：每个内部边界各 `i>j` 配对的后验概率与 `any_transition`（该边界存在转移的概率）。
- `expected_transitions`：预期段间转移数。
- `best_path`：最优分段路径（末段标 `censored: true`，长度是已见长度，不声称真实完整长度）、其 log 得分、后验概率（exp(得分-log似然)）、决胜说明与 `score_breakdown` 逐项分解。
- `top_paths`：全枚举可行时的前 5 条路径及后验概率，供复盘比较"干燥、上涨、维持、消退"各过程，保留不确定性而不是硬贴一条标签。
- `diagnostics`／`notes`：缺槽补 missing、零支持时隙、口径提醒。

## 数值处理

- 全部概率量在 log 域；`logsumexp` 用 `math.fsum` 求和，与求值顺序无关，路径得分可用 `score` 命令逐项复算到同一浮点值。
- 正态 log CDF：x>=0 用 `log1p(-0.5*erfc(x/√2))`；-38<=x<0 用 `log(0.5*erfc(-x/√2))`；x<-38 时 erfc 会下溢，切换渐近展开 Phi(x)≈phi(x)/(-x)·Σ(-1)^k(2k-1)!!x^(-2k)，该区间截断误差远低于双精度分辨率。两条分支在 -38 处连续（测试覆盖）。
- 区间概率 P(left<Y<=right) 用 log 域差 `lu + log(-expm1(ll-lu))`，不在线性空间相减；z≈7 处约 1e-12 的区间概率不丢失（测试以 erfc 尾部差作参照核对到 1e-9）。
- 两端 log CDF 在浮点下不可分辨（如区间 (41,42] 离均值数十个标准差）时，该发射项为诚实的零支持，而不是下溢造成的假不可能。
- 若没有任何合法分段能以正概率解释资料，输出 `status: uninterpretable` 与零支持时隙清单，后验表、最优路径等一律为 null——不交一张归一化失败的后验表。

## 最优路径决胜

得分并列时按约定决胜：先按逐段状态在 `state_order` 中名次的序列字典序（短序列是整个序列的前缀时短者胜），再按内部段末时隙序列字典序（边界序列不含观测窗末）。动态规划在每个结点保留最优签名；全枚举可行时以枚举的规范排序为准（`best_path.selection` 为 `enumeration`），并核对二者一致（`checks.enumeration.dp_path_matches_enumeration`）。测试覆盖：状态顺序翻转改变胜者、前缀规则、边界序列规则三方同分。

## 复算一条路径的得分

```sh
python -m response_data.cli score data/normal.json \
  '[{"state":"dry","length":2},{"state":"rising","length":2},{"state":"holding","length":2},{"state":"falling","length":1}]'
```

`score_breakdown` 给出每段的初始／转移项、持续项（中间段 P(D=d)，末段 P(D>=已见长度)）、逐时隙观测项和小计，总和用 `math.fsum` 计算，与全枚举的规范得分逐位一致；同时给出该路径的后验概率 exp(得分-log似然)。非法路径（长度超出配置最大支持、状态不相邻可转、含零支持观测）总分为 -inf（JSON 中为 null），`zero_support` 标出首个零支持位置。

## 结果边界

### 长缺测段（docs/examples/gap-long.json）

slot 0 与 slot 6 各有一条精确观测，slot 1–5 缺测。缺槽被显式补成 missing（`axis.inserted_missing_slots`），分析轴不压缩，持续长度按真实时隙计：

```
slot 1  dry 0.743 / rising 0.257
slot 2  dry 0.168 / rising 0.735 / holding 0.069 / falling 0.029
slot 3  rising 0.342 / holding 0.557 / ...
slot 4  holding 0.806 / ...
slot 5  holding 0.643 / falling 0.226 / ...
slot 6  falling 0.873 / rising 0.127
```

缺测段的发射因子为 1，后验完全由持续分布与转移结构推动，不确定性随缺口拉长而扩散：最优整路径（dry2, rising1, holding3, falling1）的后验概率只有 0.279，前 5 条路径合计也不到 0.7。边界转移后验显示转移位置本身不确定（如 dry>rising 的边界散布在 slot 0–5 之后）。这正是"几次断测不能被自动补成干燥"的口径：缺测就是缺测。

### 尾窗截断（data/right-censored-first-segment.json）

窗口只有 1 个时隙、一条精确观测 0.1。首段即末段，持续因子 P(D>=1)=1，似然等于该观测的正态密度（logL = 0.5655，密度可以大于 1）。最优路径是单段 dry、已见长度 1、`censored: true`——不声称它恰在文件最后结束，也不枚举窗口之后未观测的状态。若错误地强迫末段在窗口末结束，持续项会变成 P(D=1)=0.2，logL 被压低 log(0.2)≈-1.609；窗口越长、末段已见长度越大，两种口径差越多（例如已见长度 2 时 P(D>=2)=0.9 对 P(D=2)=0.3，见 tests 中 test_censored_tail_not_forced_end）。

### 全量程上限（data/saturated-gap.json）

slot 0 与 slot 3 是量程上限右删失读数（Y>10），slot 2 缺槽。各状态在 slot 3 的发射项对比（log）：

| 状态 | log P(Y>10)（本工具） | log 密度(10)（旧口径：当作精确值） |
| --- | --- | --- |
| dry | -1254.83 | -1249.31 |
| rising | -35.01 | -32.92 |
| holding | -75.41 | -72.23 |
| falling | -59.74 | -57.14 |

slot 3 的后验几乎全在 rising：对 holding（sd=0.5）来说 Y>10 是 12 倍标准差事件，对 rising（sd=1）只是 8 倍。饱和平台不等于"水位稳定维持"，也不等于"读数就是 10"；它只是"真实值不低于上限"的删失信息。slot 2（补入的缺测）保持真实不确定：dry 0.545 / rising 0.455，前两条路径仅在 dry>rising 边界位置上不同。旧脚本"平台即稳定、断测补干燥"的两处处理，在这里分别对应：把右删失当精确测量（发射项口径错误）、把缺测当干燥（发射因子应为 1 而非 dry 的密度）。

### 不可能持续长度与零支持

- 任何超过配置最大支持的段长（中间段 P(D=d)=0 或末段已见长度 > max D）在枚举与递推中都被排除。用 `score` 指定这样的路径，得分为 -inf 并标出位置：

  ```sh
  python -m response_data.cli score data/normal.json '[{"state":"dry","length":4},{"state":"rising","length":3}]'
  # log_score: null；zero_support: 段 0 的持续长度超出配置最大支持或概率为零
  ```

- 若资料本身没有任何合法分段能以正概率解释（docs/examples/uninterpretable.json：区间 (41,42] 对全部状态都是浮点不可分辨的零支持），输出 `status: uninterpretable`、零支持时隙清单，后验与最优路径为 null。复盘时应修正观测或参数后重跑，而不是拿到一张归一化失败的后验表。

## 设定声明

初始概率、转移矩阵、持续分布、观测均值与标准差均由配置冻结提供，不被本次资料重估。逐观测误差条件独立只是本分析的设定，用于构造可计算的似然，不推断成设备真实机理。
