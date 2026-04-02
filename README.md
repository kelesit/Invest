# CTA 量化研究系统

## 运行命令

```bash
uv run python main.py                    # 主回测（4品种组合）
uv run python tools/param_scan.py        # 参数扫描
uv run python tools/random_benchmark.py  # 随机基准检验
uv run python tools/download_data.py     # 下载数据（需 .env 中配置 DATABENTO_API_KEY）
```

## 项目结构

```
main.py                  # 入口：配置、信号路由、多品种组合回测
cta/
  data_loader.py         # Databento 连续合约加载，比例调整 + Panama 调整
  signals.py             # 信号函数（momentum, ma_crossover, combined_momentum）
  position_sizing.py     # ATR 波动率标准化仓位
  backtest.py            # 向量化回测引擎（close-to-close 模型）
  analysis.py            # 绩效指标 + 图表输出到 output/
tools/
  download_data.py       # Databento 批量下载（连续合约日线）
  param_scan.py          # lookback 参数扫描 + 训练/测试期分析
  random_benchmark.py    # 1000 次随机信号对比，计算 p-value
data/raw/                # Databento .dbn.zst 文件（ES, CL, GC, ZN, 2019-2026）
output/                  # 图表输出
```

## 关键设计决策

- **双重价格调整**：比例调整（close）用于收益率/动量/波动率计算；Panama 调整（panama_close）用于均线等价格形态信号
- **换月检测**：通过 Databento 数据中的 instrument_id 变化精确识别，不用统计启发式方法
- **仓位计算**：ATR 波动率标准化，risk_fraction 按品种数均分，使各品种风险贡献相等
- **手续费**：单边计费，turnover × commission_per_contract
- **回测模型**：close-to-close，prev_pos × 今日价格变动，信号当日收盘产生、次日生效

## 品种参数

| 品种 | point_value | commission | slippage_points |
|------|-------------|------------|-----------------|
| ES   | 50          | 2.5        | 0.25            |
| CL   | 1000        | 2.5        | 0.02            |
| GC   | 100         | 2.5        | 0.10            |
| ZN   | 1000        | 2.5        | 0.01            |
