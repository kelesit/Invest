# 量化研究系统

从 CTA 趋势跟踪入手建立量化研究基础认知，下一步转向股票截面多因子选股。

## 运行命令

```bash
# CTA 回测
uv run python main.py                    # 主回测（4品种组合）
uv run python tools/signal_compare.py    # 信号对比（夏普热力图、相关性矩阵）
uv run python tools/regime_analysis.py   # 市场环境分析
uv run python tools/signal_combination.py # 信号组合研究
uv run python tools/risk_compare.py      # 风控方案对比
uv run python tools/param_scan.py        # 参数扫描
uv run python tools/random_benchmark.py  # 随机基准检验
uv run python tools/download_data.py     # 下载数据（需 .env 中配置 DATABENTO_API_KEY）
```

## 项目结构

```
main.py                  # 入口：配置、信号路由、多品种组合回测
cta/
  data_loader.py         # Databento 连续合约加载，比例调整 + Panama 调整 + carry 数据
  signals.py             # 10 种趋势信号 + carry 因子，统一 [-1, +1] 接口，SIGNAL_REGISTRY
  position_sizing.py     # ATR 波动率标准化仓位
  backtest.py            # 向量化回测引擎（close-to-close 模型）
  analysis.py            # 绩效指标 + 图表输出到 output/
  risk.py                # 追踪止损、波动率缩放、组合波动率目标
  portfolio.py           # 等权、逆波动率权重、风险平价
tools/
  download_data.py       # Databento 批量下载（c.0 + c.1 连续合约日线）
  signal_compare.py      # 信号×品种夏普热力图、相关性矩阵、权益曲线
  regime_analysis.py     # 趋势/震荡 × 高/低波动 四种市场环境
  signal_combination.py  # 等权、逆波动率、低相关性三种组合方式
  risk_compare.py        # 7 种风控方案对比
  param_scan.py          # lookback 参数扫描 + 训练/测试期分析
  random_benchmark.py    # 1000 次随机信号对比，计算 p-value
notebooks/
  price_trend_signal_research.ipynb  # 趋势信号探索
  risk_&_portfolio.ipynb             # 风控与组合优化
  carry_analysis.ipynb               # carry 因子分析 & 多因子组合
docs/
  price_trend_signal_principles.md   # 信号原理（经典定义 vs 项目实现）
  risk_principles.md                 # 风控原理
  carry_principles.md                # carry 因子原理
data/raw/                # Databento .dbn.zst 文件（ES, CL, GC, ZN, 2019-2026）
output/                  # 图表输出
```

## 关键设计决策

- **双重价格调整**：比例调整（close）用于收益率/动量/波动率计算；Panama 调整（panama_close）用于均线等价格形态信号；carry 因子用原始未调整价格
- **换月检测**：通过 Databento 数据中的 instrument_id 变化精确识别
- **信号接口**：所有信号统一输出 [-1, +1]，通过 SIGNAL_REGISTRY 注册管理
- **仓位计算**：ATR 波动率标准化，risk_fraction 按品种数均分
- **回测模型**：close-to-close，prev_pos × 今日价格变动，信号当日收盘产生、次日生效

## 品种参数

| 品种 | point_value | commission | slippage_points |
|------|-------------|------------|-----------------|
| ES   | 50          | 2.5        | 0.25            |
| CL   | 1000        | 2.5        | 0.02            |
| GC   | 100         | 2.5        | 0.10            |
| ZN   | 1000        | 2.5        | 0.01            |
