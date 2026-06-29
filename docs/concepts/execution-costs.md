# 成本、滑点与 `tr_close`

> status: active
> owner: portfolio-backtester
> last_verified: 2026-06-29
> source_of_truth: yes
> superseded_by: n/a

本页解决什么：把仓库当前的交易成本、滑点、`tr_close` 和分红处理假设放到一页里。\
范围：broker 级 TCA 和真实现金账本需要单独实现。\
适合谁：正在看 HK selected 回测、想知道结果里哪些是执行近似，哪些是总回报价格代理的读者。\
读完你会得到什么：当前算法、数据依赖、适用范围和已知限制。\
相关页面：`strategy-pipeline/docs/config.md`、`strategy-pipeline/docs/outputs.md`、`cstree.backtesting.execution`、`market_data_platform.data_providers`

迁移说明：本页从 `strategy-pipeline/docs/concepts/execution-costs.md` 迁入。回测成本、滑点和执行模拟由 `portfolio-backtester` 维护；配置合成和 CLI 编排仍由 `strategy-pipeline` 维护。

## 先说结论

仓库当前提供的是两层交易成本建模：

1. 默认研究线常用 `transaction_cost_bps`，这是单一平面成本近似。
2. `backtest.execution` 提供更细的 execution 结构：买卖分边费用、固定滑点、participation 滑点、开平仓价列和最小流动性约束。

这已经比简易回测更进一步。相较最完整的机构级别 TCA，目前缺失的包括：

* 用真实成交回报去校准参数。
* 在需要时引入更细市场数据。

## 当前成本与滑点是怎么工作的

`backtest.execution` 目前由三部分组成：

* `entry_policy` / `exit_policy`：决定回测用哪一列价格做成交价，例如 `open`、`close`、`tr_close`。
* `cost_model`：显式费用模型，支持统一 `bps`、分边 `side_bps`、或关闭。
* `slippage_model`：支持 `none`、固定 `bps`、或按参与率估计的 `participation`。

`participation` 现在的近似形式是：

* 用 `trade_weight * portfolio_value / amount_col` 估计成交额占当日流动性的比例。
* 在这个比例上叠加 `base_bps` 和 `impact_bps * participation^power`。

所以目前的状态是容量 / 冲击 stress model。

如果需要回答目标仓位能否在给定成交量约束下买入或卖出，可以另外启用 `backtest.execution_sim`。它保留主回测净值，并把 `positions_by_rebalance.csv` 当成目标订单，按 `participation_rate * min(liquidity_cols)` 做日度成交容量模拟；未完成买入保留现金，未完成卖出保留旧仓并标记 delayed sell。启用后还会生成 `execution_sim_executed_daily.csv`，提供容量约束和未成交现金 / 持仓状态进入净值后的日度结果。

如果 panel 同时包含 `is_buy_tradable` 与 `is_sell_tradable`，容量模拟会按方向分别使用：
涨停可以阻断买入而不阻断卖出，跌停可以阻断卖出而不阻断买入。`is_tradable` 仍作为
保守兼容列供单列 backtest 接口使用。完整 A 股撮合还需要覆盖 T+1 可卖数量、板块差异
和券商侧拒单仍需要单独证据。

历史月频 HK execution 配置已经迁入 `strategy-pipeline/docs/archive/research/hk/README.md` 归档入口。active
`configs/experiments/` 不再保留这些港股执行层配置；当前主线新增实验应先围绕 A 股 default
或显式 A 股候选配置建立 execution evidence。

## 为什么现在建议用 `adv20_amount`

如果 `entry_policy.price_col=open`，而 `slippage_model.amount_col=amount` 或 `constraints.amount_col=amount`，那就会拿同一天的日成交额去约束开盘成交。

这在工程上常见，但在严格意义上存在轻微 pre-trade 不可知问题，因为开盘时你并不知道全天总成交额。

仓库现在支持把流动性代理列直接写成滞后版本，例如：

* `adv20_amount`
* `medadv20_amount`

它们表示按 symbol 计算、排除当日后的过去 `20` 个交易日平均或中位成交额。对日线级回测，这是比同日 `amount` 更稳妥的默认值。

## `tr_close` 是什么

`tr_close` 在仓库里表示总回报价格代理。

当前实现依赖下面几条路之一：

* 本地 `ex_factors_dir`
* 在线 provider 的 `adjust_type=pre/post`
* 或 daily 源数据里已经自带 `tr_close`

简化说，代码会把 `close` 结合复权因子变成 `tr_close`，让价格类特征、标签和回测一起走 total-return 口径。

现在仓库还会把 `tr_close` 的来源沉到 `summary.json -> data -> price_col_diagnostics`。  
如果你配置了本地 `ex_factors_dir`，但某些 symbol 没有对应 ex-factor 行，run 日志会显式提示；summary 里也能看到到底是：

* `local_ex_factors`
* `provider_adjusted_price`
* `input_frame`
* `input_frame_missing_ex_factors`
* `close_fallback_missing_ex_factors`

这意味着：

* 如果你的目标是低频研究、比较信号强弱、避免分红导致价格跳空污染收益标签，那么 `tr_close` 是合理的。
* 如果你的目标是重建真实现金分红到账、再投资时点、税率、到账日差异，那么 `tr_close` 不够。

## 目前距离最严谨的考虑现金分红影响的回测还缺了什么

现金分红账本至少还差这些维度：

* `payable_date` 而不只是 `ex_dividend_date`
* 税前/税后金额
* 持仓股数与 round lot 影响
* 分红现金是否再投资、何时再投资
* 多市场税率和券商处理差异

仓库现有 `dividends` 资产更适合做核对或后续扩展。

## 当前实现的适用边界

以下场景里，当前做法通常已经够用：

* 港股低频、月频或季频、Top-K 选股研究
* 主要关心信号方向、组合排序和大体成本拖累
* 希望 `close` 和 total-return 口径能快速做 A/B

以下场景里，它就不够了：

* 你开始关心开盘成交是否真的可行
* 你需要盘中 VWAP/TWAP 偏离
* 你在做集合竞价、盘口冲击或 broker 级 TCA
* 你要做真实现金分红账本

`backtest.execution_sim` 可以覆盖其中一部分容量边界和分批执行问题，但仍属于日线级容量模拟，逐笔撮合器需要单独实现。

## 当前 HK `5m` 校准报告还要再记一个口径

当前工作区里的 HK `5m` 缓存是 provider 默认 `adjust_type=pre` 的价格序列。  
因此，如果直接拿 `amount / volume` 去对比历史复权后的 `open`，长期样本上的 `VWAP` 会失真。

所以仓库里的 intraday 滑点报告当前采用的是：

* `vwap_method=bar_price_volume_proxy`
* 用每根 `5m` bar 的 `OHLC` 均价按 bar `volume` 加权，近似 session price center

这能让现有缓存继续用于经验校准，但要明确：

* 它是可复用的研究 proxy，距离tick 级真实 VWAP仍有距离
* 它当前用于离线校准 execution 参数，日线 backtest 的逐 bar 撮合输入需要单独设计
* 如果后面要做更严肃的盘中执行研究，优先考虑重新下载 `adjust_type=none` 的分钟线，或直接引入更细成交数据

## 推荐的下一步

优先级从高到低通常是：

1. 固定 `ex_factors / dividends / shares` 这组轻量原料层。
2. 用真实成交回报校准 `buy_bps / sell_bps / base_bps / impact_bps`。
3. 回测层优先用 `adv20_amount` 或 `medadv20_amount`，避免 `open + same-day amount`。
4. 若只是先把研究线做对，优先在当前 active baseline 上显式写清 execution config；单一的 `transaction_cost_bps` 已经不再适用。
   需要复现港股 total-return execution 口径时，从 archive README 找到历史配置文件名和对应 run 产物。
5. 如果误差主要来自执行假设，再补精选池或全市场的 `5m` 数据做经验滑点校准。
6. 只有在策略明显进入容量或盘中执行问题时，才考虑更细的 `1m`、tick 或盘口数据。

## 最后一句

`tr_close`、execution cost model、现金分红账本，是三件相邻但不同的事。把这三件事写清楚，比再多下载一批数据更重要。
