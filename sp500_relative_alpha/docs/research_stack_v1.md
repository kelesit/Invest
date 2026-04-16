# Research Stack v1

这份文档定义这个项目采用的**正式研究分层**。

目标不是让结构看起来完整，而是强制我们先回答：

- 我们到底在主张什么关于市场的事实
- 什么观察会推翻这个主张
- 什么结果才配被叫作“成功”

在这套分层里，`L0-L3` 冻结之前，`L4` 的实现讨论一律不具备正式地位。

---

## L0. 研究声明（Epistemic Claim）

系统如果工作，究竟在主张什么关于世界的事实。

必须明确：

- claim 的类型
  - empirical claim
  - methodological claim
  - impossibility / null-result claim
- claim 的可证伪形式
  - 什么观察会让该 claim 被否定
- claim 的可重复性要求
  - 只在某段时间成立是否算数
  - 是否要求跨时间、跨实现仍然成立

补充规则：

- empirical claim 和 methodological claim 必须分开写
- 不允许在结果不好时把 empirical claim 偷换成“我只是想验证 pipeline”
- null result 也是正式结果

---

## L1. 研究问题的精确形式

把 L0 的 claim 翻译成可操作问题：

> 在 `[universe 的理论性质]` 内，  
> 在 `[时间范围的理论约束]` 内，  
> 使用 `[输入空间的理论限制]`，  
> 构造 `[输出类型的理论定义]`，  
> 评估它在 `[评估协议的理论要求]` 下是否满足 `[成功 / 证伪条件]`。

规则：

- 每个括号先定义“性质”，再填具体值
- 在这一层，指数名、供应商名、目录名都不是核心
- 如果一句研究问题里写不出成功条件和证伪条件，它还不是一个正式问题

---

## L2. 核心对象定义

这一层只回答：每个对象**必须满足什么**。

### L2a. Universe

必须定义：

- point-in-time 可重构性
- survivorship bias 处理
- 流动性下限及其理论理由
- 可交易性约束
- 代表性要求
- 时间稳定性
- identity model
  - 证券主键必须能够跨 ticker 变化保持连续

### L2b. Label

必须定义：

- 数学定义
- 时间对齐规则
- 残差空间
- 经济可解释性
- 可交易性证明
- noise floor 的理论估计
- horizon 的理论依据

### L2c. 输入空间（Feature Space）

必须定义：

- 允许的原始数据类型
- point-in-time 信息边界
- 变换族的封闭性
- 禁止的 feature 类
- 可复现性要求
- cross-sectional 与 time-series 构造的边界

### L2d. 评估协议

必须定义：

- out-of-sample 的精确定义
- 时间结构约束
- 主 metric 与 noise floor 估计
- 多重检验协议
- anti-p-hacking 纪律
- 再现性要求

### L2e. 成功与证伪条件

必须定义：

- 成功的定量门槛
- 放弃的定量门槛
- 中间状态的处理
- 成功与失败的对称性

### L2f. Benchmark / Counterfactual

必须定义：

- 最弱基准
  - 随机排序
  - constant scorer
- 统计基准
  - 线性模型
  - 简单单因子排序
- 经济基准
  - benchmark passive 持有
  - naive top-k / equal-weight
- 实现基准
  - 不同实现者 / 不同代码路径是否得出同结论

没有 benchmark，所谓“有效”很容易只是打赢了一个稻草人。

---

## L3. 系统级约束

这些约束独立于 L2 的具体取值。

必须定义：

- anti-snooping 纪律的数量化形式
- 预注册协议
- 平稳性假设的显式承认
- 样本预算
- 可重放性（replay）要求
- 研究过程中的信息隔离

规则：

- OOS 不是免费资源
- 每一次看最终检验集，都要消耗研究信用
- 任何无法完整 replay 的结论，都不应被长期相信

---

## L4. 实现层

只有到这里，才允许讨论：

- 数据 / 供应商选择
- 编程语言 / 工具链
- 代码目录组织
- 测试方案
- 协作规范

在此之前，任何 L4 讨论都只是“实现冲动”，不是正式设计。

---

## 推荐推进顺序

`L0 -> L1 -> L2a -> L2b -> L2c -> L2d -> L2e -> L2f -> L3 -> L4`

说明：

- `L2` 内部天然有循环
- universe 会约束 label
- label 会约束评估协议
- 输入空间会约束可实现的 label 与 horizon

因此实际推进应是：

1. 先给出 `v0`
2. 检查内部矛盾
3. 迭代到自洽
4. 自洽后再进入实现层
