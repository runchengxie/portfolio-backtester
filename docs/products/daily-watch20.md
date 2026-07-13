# DailyWatch20 产品模块

DailyWatch20 根据调用方预先计算的模型分数和守卫因子，构造每日 4+16 观察名单。它属于具体产品规则，位于 `cstree.backtesting.products` 命名空间，不是通用回测内核的一部分。

## 依赖边界

DailyWatch20 只使用 NumPy 和 Pandas 处理输入截面。它不会训练模型，也不导入 scikit-learn 或 XGBoost。输入中的 `xgb_score` 只是默认分数列名称，模型训练和预测由调用方负责。

导入 `cstree.backtesting` 不会加载产品模块。只有显式导入产品命名空间或访问顶层兼容名称时才会加载 DailyWatch20。

## 推荐导入

```python
from cstree.backtesting.products import (
    DailyWatch20Config,
    GuardFactorSpec,
    select_daily_watch20,
)
```

`DailyWatch20Config`、`GuardFactorSpec` 和 `select_daily_watch20` 仍可从 `cstree.backtesting` 顶层导入，以兼容已有调用方。顶层名称采用延迟加载，不会扩大核心导入边界。

## 旧模块路径

下面的历史路径继续提供同一组对象：

```python
from cstree.backtesting.daily_watch20 import select_daily_watch20
```

该模块现在是弃用兼容层，导入时发出 `DeprecationWarning`。新代码应使用 `cstree.backtesting.products`。已有序列化对象仍可通过旧模块路径找到对应公开名称。

## 输入与输出

输入是单个交易日的截面 `DataFrame`，至少包含：

- 交易日期
- 证券代码
- 行业
- 已计算的模型分数
- 硬性可用标记
- 一个或多个守卫因子

返回的 `DailyWatch20Result` 包含观察名单和选择回执。具体字段、约束和失败关闭行为由 `DailyWatch20Config` 及测试样例定义。
