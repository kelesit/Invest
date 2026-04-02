"""批量下载 Databento 期货日线数据（连续合约符号）。

用法: uv run python tools/download_data.py

需要设置环境变量 DATABENTO_API_KEY，或在下方直接填入。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import databento as db

# ============================================================
# 配置
# ============================================================
load_dotenv()
API_KEY = os.environ.get("DATABENTO_API_KEY", "")
OUTPUT_DIR = Path("data/raw")
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1d"

START = "2019-01-01"
END = "2026-03-31"

# 品种：使用连续合约符号 {root}.c.0 = 主力合约
PRODUCTS = {
    "ES": "E-mini S&P 500",
    "CL": "Crude Oil",
    "GC": "Gold",
    "ZN": "10-Year Treasury Note",
}


def build_download_list() -> list[dict]:
    """生成下载任务列表：每个品种一个请求。"""
    tasks = []
    for root, name in PRODUCTS.items():
        symbol = f"{root}.c.0"
        filename = f"{root}-continuous-{SCHEMA}-{START}_{END}.dbn.zst"
        filepath = OUTPUT_DIR / filename

        tasks.append({
            "symbol": symbol,
            "root": root,
            "name": name,
            "filepath": filepath,
        })
    return tasks


def download_all():
    if not API_KEY:
        print("错误: 请设置环境变量 DATABENTO_API_KEY")
        print("  export DATABENTO_API_KEY=db-xxxxx")
        return

    client = db.Historical(API_KEY)
    tasks = build_download_list()

    print(f"共 {len(tasks)} 个品种待下载")
    print(f"品种: {[t['root'] for t in tasks]}")
    print(f"时间: {START} ~ {END}")
    print(f"Schema: {SCHEMA}")
    print()

    # 先估算费用
    print("正在估算费用...")
    total_cost = 0.0
    valid_tasks = []

    for task in tasks:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[task["symbol"]],
                stype_in="continuous",
                schema=SCHEMA,
                start=START,
                end=END,
            )
            print(f"  {task['root']:<4} ({task['name']}): ${cost:.2f}")
            total_cost += cost
            valid_tasks.append(task)
        except Exception as e:
            print(f"  {task['root']:<4} 跳过: {e}")

    print(f"\n预估总费用: ${total_cost:.2f}")
    print(f"有效品种: {len(valid_tasks)}/{len(tasks)}")

    confirm = input("\n确认下载? (y/N): ")
    if confirm.lower() != "y":
        print("已取消")
        return

    # 开始下载
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = []

    for task in valid_tasks:
        filepath = task["filepath"]

        if filepath.exists():
            print(f"  跳过 {task['root']} (已存在)")
            downloaded += 1
            continue

        try:
            print(f"  下载 {task['root']} ({task['name']})...", end=" ", flush=True)
            client.timeseries.get_range(
                dataset=DATASET,
                symbols=[task["symbol"]],
                stype_in="continuous",
                schema=SCHEMA,
                start=START,
                end=END,
                path=str(filepath),
            )
            print("OK")
            downloaded += 1
        except Exception as e:
            print(f"失败: {e}")
            failed.append(task["root"])

    print(f"\n完成: {downloaded} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {failed}")


if __name__ == "__main__":
    download_all()
