# 持仓输出约定

本页说明 `positions_by_rebalance.csv` 的稳定字段、校验行为和下游使用方式。

## 标准文件名

组合构造结果的标准文件名为：

```text
positions_by_rebalance.csv
```

对应的契约名称为 `portfolio_backtester.positions_by_rebalance`，当前版本为 `1`。

## 必需字段

| 字段 | 类型要求 | 含义 |
| --- | --- | --- |
| `rebalance_date` | 可以解析为日期 | 目标持仓对应的调仓日期 |
| `symbol` | 非空字符串 | 证券代码 |
| `weight` | 可以转换为数值 | 目标权重 |

日期可以使用 `YYYYMMDD`、`YYYY-MM-DD` 或其他可以由 pandas 解析的格式。为了方便跨系统交换，建议统一使用 `YYYYMMDD`。

## 常用可选字段

| 字段 | 含义 |
| --- | --- |
| `entry_date` | 计划建仓日期 |
| `side` | 持仓方向，例如 `long` |
| `signal` | 构造持仓时使用的分数 |
| `rank` | 证券在当期候选集中的排序 |

专用组合构造器可以增加额外字段，例如组合分组、主题、行业或模型版本。下游读取程序应保留未知字段，并只依赖已经约定的字段。

## 示例

```csv
rebalance_date,entry_date,symbol,weight,side,signal,rank
20260102,20260105,000001.SZ,0.40,long,1.25,1
20260102,20260105,600000.SH,0.35,long,1.10,2
20260102,20260105,000002.SZ,0.25,long,0.98,3
```

## 当前校验范围

`validate_positions_by_rebalance_frame` 会检查：

- 三个必需字段是否存在
- `rebalance_date` 是否可以解析为日期
- `symbol` 是否为空
- `weight` 是否可以转换为数值

当前契约不会自动检查：

- 同一日期和证券是否重复
- 单期权重和是否等于 1
- 权重是否为负数
- `side` 是否只包含允许值
- `entry_date` 是否晚于调仓日期
- 证券代码是否符合特定市场格式

这些规则取决于组合类型和调用方需求。需要严格约束时，应在生成持仓的模块中补充校验，并为规则添加测试。

可以使用以下函数进行校验：

```python
from portfolio_backtester import (
    assert_positions_by_rebalance_frame,
    validate_positions_by_rebalance_frame,
)

issues = validate_positions_by_rebalance_frame(positions)
assert_positions_by_rebalance_frame(positions)
```

`validate_positions_by_rebalance_frame` 返回问题列表。`assert_positions_by_rebalance_frame` 在发现问题时抛出 `ValueError`。

## 持仓回放时的处理

`run_position_backtest` 会先应用基础契约校验，然后执行以下处理：

- 把 `symbol` 转为字符串
- 把 `weight` 转为数值
- 如果存在 `side`，只保留值为 `long` 的行
- 只保留正权重持仓
- 同一调仓日的重复证券会在回放阶段按证券合计权重
- 默认把有效权重重新归一化
- `preserve_gross_exposure=True` 时允许保留现金权重

持有期由单独的 `periods` 表决定。该表至少需要 `rebalance_date`、`entry_date` 和 `exit_date`。持仓文件中的 `entry_date` 主要用于记录和跨系统交换，回放时仍以 `periods` 表为准。

## 价格缺失

某个持有期内缺少开仓价或退出价时，回放会移除缺少有效价格的证券。默认情况下，剩余持仓权重会重新归一化。

这种处理会改变实际参与回测的证券数量。分析结果时应检查：

- `missing_price_count`
- `position_count`
- `gross_exposure`
- `cash_weight`

需要保留未投资现金时，可以设置：

```python
PositionBacktestConfig(preserve_gross_exposure=True)
```

## 其他文件名

`positions_current.csv`、`trades_by_rebalance.csv` 和 `latest.json` 可能出现在上层任务编排或发布流程中。它们目前不属于本包的稳定输出契约。

公共集成应优先依赖 `positions_by_rebalance.csv` 和契约校验函数。上层项目可以定义自己的快照、交易差异和便利指针，但应单独记录字段和更新规则。

## 兼容性建议

- 新增可选字段通常可以保持向后兼容
- 删除或改名必需字段会破坏契约
- 修改日期含义、权重含义或重复行处理方式时，应升级契约版本
- 写出文件前保持列名稳定
- 保存生成持仓所使用的模型版本和参数
- 在 PR 中说明对下游读取程序的影响
