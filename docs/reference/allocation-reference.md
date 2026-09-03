# 执行分配参考资产

`portfolio_backtester.allocation_reference` 读取执行侧使用的价格和整手信息，并把它们安全地合并到组合选择结果中。

## 输入格式

参考文件支持 CSV、TXT、JSON、JSONL 和 Parquet，必须包含以下字段：

- `symbol`
- `price`
- `round_lot`
- `price_date`

所有行必须属于同一个 `price_date`。`price` 和 `round_lot` 必须为正数，`round_lot` 必须是整数，`symbol` 不能重复。缺少 `order_book_id` 时，模块会使用 `symbol` 作为默认值。

## 使用方式

```python
from portfolio_backtester.allocation_reference import (
    join_allocation_reference,
    load_allocation_reference,
)

reference = load_allocation_reference("reference.csv")
ready = join_allocation_reference(selection, reference)
```

合并操作保留选择结果的顺序。选择结果中找不到价格或整手信息的证券会直接失败，避免执行层生成不完整的数量。

持仓筛选由 `portfolio_backtester.allocation_selection` 提供。它从保存的持仓文件中选取指定日期前最新的一期数据，按方向和排名完成 Top-N 筛选，不读取策略配置或运行时目录。

分配结果的文本表格和 CSV 输出由 `portfolio_backtester.allocation_rendering` 负责。表格格式会按中英文字符宽度对齐，CSV 输出保留 DataFrame 的列顺序。

## 职责边界

本模块只负责执行所需的参考资产读取、校验和连接。它不生成行情、不决定选股结果，也不负责订单发送。数据来源和快照日期由调用方提供并记录在运行产物中。
