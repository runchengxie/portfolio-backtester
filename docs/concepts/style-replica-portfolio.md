# 风格复制组合构造器

`style_replica_portfolio` 提供一个双组合腿的专用构造器。它适合已有 `score_a`、`score_b`、主题标签和行业标签的日频信号。

通用 Top-K 组合优先使用 `StrategySpec` 和 `construct_positions_from_strategy`。本页介绍的构造器带有固定的 A 组合腿和 B 组合腿规则，适用范围更窄。

## 输入字段

输入使用长表格式，每行代表一个日期和一个证券。

| 字段 | 要求 | 含义 |
| --- | --- | --- |
| `signal_date` 或 `trade_date` | 必需 | 信号日期 |
| `symbol` | 必需 | 证券代码 |
| `score_a` | A 组合腿需要 | A 组合腿排序分数 |
| `score_b` | B 组合腿需要 | B 组合腿排序分数 |
| `theme` | A 组合腿需要 | 主题分组 |
| `industry` | B 组合腿建议提供 | 行业分组 |
| `leg` | 可选 | 上游提供的组合腿标签，当前构造过程不会依赖该列 |

缺少 `score_a`、`score_b`、`theme` 或 `industry` 时，函数会补充空列。缺少日期列或 `symbol` 时，输入无法正常构造持仓。

## A 组合腿

A 组合腿按主题分别选股：

1. 在每个主题内按 `score_a` 从高到低排序
2. 新持仓需要进入主题配额范围
3. 原有持仓可以在缓冲区内继续保留
4. 每个主题最多保留 `theme_quotas` 指定的数量
5. 初始选择结束后，构造器会尝试用剩余主题证券补足 `a_slots`

`a_buffer_exit_multiplier` 控制原有持仓的退出缓冲区。例如主题配额为 10，倍率为 1.3 时，原有持仓排名降到约 13 名之后才会退出初始保留范围。

## B 组合腿

B 组合腿按 `score_b` 从高到低排序，并在初始选择阶段应用：

- `b_slots` 指定的目标数量
- `b_industry_cap` 指定的单一行业数量上限
- `b_buffer_entry_rank` 指定的新持仓进入范围
- `b_buffer_exit_rank` 指定的原有持仓保留范围
- `b_max_daily_replacements` 指定的每日新增数量上限

初始选择不足 `b_slots` 时，代码会放宽排名范围。随后还会从剩余证券中继续补足数量。补足阶段没有再次应用行业数量上限和每日新增数量上限，因此实际结果可能超过这两项设置。使用者需要在下游校验持仓。

## 重叠持仓

`overlap_policy` 支持两种方式：

- `aggregate` 将同时入选 A 和 B 的证券合并，权重最高为 `max_name_weight`
- `deduplicate` 从 B 组合腿移除与 A 组合腿重复的证券

每个普通槽位使用 `normal_slot_weight`。构造器不会自动把全部权重归一化到 1。实际总权重取决于槽位数量、重叠数量和权重设置。

## 输出字段

`build_style_replica_positions` 返回的持仓表包含：

- `rebalance_date`
- `entry_date`
- `symbol`
- `weight`
- `side`
- `leg`
- `signal`
- `score_a`
- `score_b`
- `theme`
- `industry`
- `rank`

输出满足 `positions_by_rebalance.csv` 的必需字段约定。

## 辅助函数

`compute_daily_changes` 比较相邻日期的持仓，输出 `new`、`exit`、`weight_change` 和 `stay`。

`compute_style_exposure_summary` 汇总单日证券数量、组合腿数量、总权重、主题分布和行业分布。

`compute_daily_exposure` 对全部调仓日期逐日调用暴露汇总函数。

## 当前限制

以下配置字段目前没有参与持仓计算：

- `a_capital_weight`
- `b_capital_weight`
- `max_daily_replacements`
- `model_version`

其中 A 和 B 的实际资金比例由 `normal_slot_weight`、槽位数量和重叠处理决定。

该模块包含专用策略命名和固定双组合腿语义。公开 API 已导出这些函数，但兼容性和约束强度低于通用 `StrategySpec` 路径。修改规则时应增加专门的行为测试，并同步更新本页。