#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "uploaded_stock_indicator_analysis"
CHART_DIR = OUTPUT_DIR / "charts"
DATA_DIR = OUTPUT_DIR / "data"

DEFAULT_INPUTS = [
    {
        "label": "三一重工",
        "expected_code": "600031.SH",
        "path": Path("/Users/oir/Downloads/%E4%B8%89%E4%B8%80%E9%87%8D%E5%B7%A5%E8%A1%8C%E6%83%85%E6%95%B0%E6%8D%AE.csv"),
    },
    {
        "label": "平安集团文件",
        "expected_code": "601318.SH",
        "path": Path("/Users/oir/Downloads/%E5%B9%B3%E5%AE%89%E9%9B%86%E5%9B%A2%E8%A1%8C%E6%83%85%E6%95%B0%E6%8D%AE.csv"),
    },
]

REQUIRED_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def parse_input_spec(value):
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = unquote(Path(path).stem)
    return {
        "label": label.strip() or unquote(Path(path).stem),
        "expected_code": "",
        "path": Path(path).expanduser(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="诊断股票行情 CSV，计算 RSI、MACD、布林带和 ATR，并输出图表与报告。"
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="输入 CSV，可重复传入。格式支持 path 或 label=path。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="输出目录，默认 reports/uploaded_stock_indicator_analysis。",
    )
    return parser.parse_args()


def read_csv(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 {path}: {last_error}")


def prepare_data(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def wilder_average(series, period):
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    if len(values) < period + 1:
        return pd.Series(out, index=series.index)
    first = np.nanmean(values[1 : period + 1])
    out[period] = first
    for idx in range(period + 1, len(values)):
        out[idx] = (out[idx - 1] * (period - 1) + values[idx]) / period
    return pd.Series(out, index=series.index)


def add_indicators(df):
    out = df.copy()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_average(gain, 14)
    avg_loss = wilder_average(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100
    out.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd_dif"] = out["ema_12"] - out["ema_26"]
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = 2 * (out["macd_dif"] - out["macd_dea"])

    out["bb_mid_20"] = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std(ddof=1)
    out["bb_upper_20_2"] = out["bb_mid_20"] + 2 * bb_std
    out["bb_lower_20_2"] = out["bb_mid_20"] - 2 * bb_std
    out["bb_width_20_2"] = (out["bb_upper_20_2"] - out["bb_lower_20_2"]) / out["bb_mid_20"]
    out["bb_percent_b_20_2"] = (out["close"] - out["bb_lower_20_2"]) / (
        out["bb_upper_20_2"] - out["bb_lower_20_2"]
    )

    prev_close = out["close"].shift(1)
    tr_parts = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["true_range"] = tr_parts.max(axis=1)
    out["atr_14"] = wilder_average(out["true_range"], 14)
    out["atr_pct_14"] = out["atr_14"] / out["close"] * 100
    return out


def diagnose(df, meta):
    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    unique_codes = sorted(df["ts_code"].dropna().astype(str).unique().tolist())
    diagnostics = {
        "label": meta["label"],
        "file_name": unquote(meta["path"].name),
        "path": str(meta["path"]),
        "encoding": meta["encoding"],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_ts_code": unique_codes,
        "expected_code": meta.get("expected_code"),
        "start_date": df["date"].min().date().isoformat(),
        "end_date": df["date"].max().date().isoformat(),
        "duplicate_trade_date_count": int(df["trade_date"].duplicated().sum()),
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "negative_numeric_counts": {col: int((df[col] < 0).sum()) for col in numeric_cols},
        "price_logic_violations": {
            "high_lt_low": int((df["high"] < df["low"]).sum()),
            "high_lt_open_or_close": int(((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()),
            "low_gt_open_or_close": int(((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()),
        },
    }
    diagnostics["code_warning"] = (
        bool(meta.get("expected_code") and unique_codes and meta["expected_code"] not in unique_codes)
    )
    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isna().sum()
    desc["skew"] = df[numeric_cols].skew(numeric_only=True)
    desc["kurtosis"] = df[numeric_cols].kurt(numeric_only=True)
    return diagnostics, desc


def make_chart(df, label, chart_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, gridspec_kw={"height_ratios": [2.1, 1, 1.2, 1]})
    code = str(df["ts_code"].iloc[0]) if "ts_code" in df.columns and len(df) else label
    fig.suptitle(f"{code} Technical Indicators", fontsize=16, fontweight="bold")

    axes[0].plot(df["date"], df["close"], label="Close", color="#2563eb", linewidth=1.6)
    axes[0].plot(df["date"], df["bb_mid_20"], label="BB Mid 20", color="#4b5563", linewidth=1.1)
    axes[0].plot(df["date"], df["bb_upper_20_2"], label="BB Upper", color="#16a34a", linewidth=1.1)
    axes[0].plot(df["date"], df["bb_lower_20_2"], label="BB Lower", color="#dc2626", linewidth=1.1)
    axes[0].fill_between(
        df["date"].to_numpy(),
        df["bb_lower_20_2"].to_numpy(dtype=float),
        df["bb_upper_20_2"].to_numpy(dtype=float),
        color="#93c5fd",
        alpha=0.18,
    )
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", ncol=4)

    axes[1].plot(df["date"], df["rsi_14"], label="RSI 14", color="#7c3aed", linewidth=1.3)
    axes[1].axhline(70, color="#dc2626", linestyle="--", linewidth=0.9)
    axes[1].axhline(30, color="#16a34a", linestyle="--", linewidth=0.9)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("RSI")
    axes[1].legend(loc="upper left")

    axes[2].plot(df["date"], df["macd_dif"], label="DIF", color="#2563eb", linewidth=1.2)
    axes[2].plot(df["date"], df["macd_dea"], label="DEA", color="#f59e0b", linewidth=1.2)
    colors = np.where(df["macd_hist"] >= 0, "#16a34a", "#dc2626")
    axes[2].bar(df["date"], df["macd_hist"], label="MACD Hist", color=colors, alpha=0.55, width=1)
    axes[2].axhline(0, color="#111827", linewidth=0.8)
    axes[2].set_ylabel("MACD")
    axes[2].legend(loc="upper left", ncol=3)

    axes[3].plot(df["date"], df["atr_14"], label="ATR 14", color="#92400e", linewidth=1.3)
    axes[3].set_ylabel("ATR")
    axes[3].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def safe_slug(text):
    table = {
        "三一重工": "sany_600031_SH",
        "平安集团文件": "pingan_file_000001_SZ",
    }
    return table.get(text, text.lower().replace(" ", "_"))


def write_report(diagnostics, desc_paths, indicator_paths, chart_paths, latest_rows, report_path):
    source_urls = [
        "https://www.investopedia.com/terms/r/rsi.asp",
        "https://www.investopedia.com/terms/m/macd.asp",
        "https://www.investopedia.com/terms/b/bollingerbands.asp",
        "https://en.wikipedia.org/wiki/Average_true_range",
    ]
    lines = [
        "# 两只股票行情数据诊断与技术指标分析",
        "",
        "生成日期：2026-07-03",
        "",
        "## 1. 数据基础诊断",
        "",
    ]
    for item in diagnostics:
        lines += [
            f"### {item['label']}",
            "",
            f"- 文件名：`{item['file_name']}`",
            f"- 路径：`{item['path']}`",
            f"- 编码：`{item['encoding']}`",
            f"- 行数：{item['rows']}",
            f"- 数据区间：{item['start_date']} 至 {item['end_date']}",
            f"- 数据代码：{', '.join(item['unique_ts_code'])}",
            f"- 缺失值总数：{sum(item['missing_values'].values())}",
            f"- 重复交易日数量：{item['duplicate_trade_date_count']}",
            f"- 价格逻辑异常：{item['price_logic_violations']}",
        ]
        if item["code_warning"]:
            lines.append(
                f"- 数据一致性提醒：文件标签期望代码 `{item['expected_code']}`，但实际数据代码为 `{', '.join(item['unique_ts_code'])}`。请确认文件名和股票代码是否匹配。"
            )
        lines.append("")

    lines += [
        "描述性统计量已输出为 CSV：",
        "",
    ]
    for label, path in desc_paths.items():
        lines.append(f"- {label}: `{path}`")
    lines += [
        "",
        "## 2. 指标含义、计算方法与作用",
        "",
        "### RSI",
        "",
        "RSI 是动量震荡指标，用于衡量近期上涨幅度与下跌幅度的相对强弱。本文采用 Wilder 风格的 14 日平滑：",
        "",
        "- `delta = close.diff()`",
        "- `gain = max(delta, 0)`",
        "- `loss = max(-delta, 0)`",
        "- `avg_gain` 与 `avg_loss` 使用 Wilder 平滑",
        "- `RS = avg_gain / avg_loss`",
        "- `RSI = 100 - 100 / (1 + RS)`",
        "",
        "常见作用：辅助观察超买/超卖、强弱变化和动量背离。常用阈值为 70 和 30，但不应单独作为买卖依据。",
        "",
        "### MACD",
        "",
        "MACD 用长短周期 EMA 的差值观察趋势和动能。本文采用标准参数 `12, 26, 9`：",
        "",
        "- `EMA12 = EMA(close, 12)`",
        "- `EMA26 = EMA(close, 26)`",
        "- `DIF = EMA12 - EMA26`",
        "- `DEA = EMA(DIF, 9)`",
        "- `MACD Histogram = 2 * (DIF - DEA)`",
        "",
        "常见作用：观察趋势方向、动能强弱、DIF/DEA 交叉和柱状图扩张或收缩。",
        "",
        "### 布林带 Bollinger Bands",
        "",
        "布林带用移动平均和标准差刻画价格相对波动区间。本文采用 `20 日 SMA ± 2 倍标准差`：",
        "",
        "- `mid = SMA(close, 20)`",
        "- `std = rolling_std(close, 20)`",
        "- `upper = mid + 2 * std`",
        "- `lower = mid - 2 * std`",
        "- `bandwidth = (upper - lower) / mid`",
        "- `%B = (close - lower) / (upper - lower)`",
        "",
        "常见作用：观察价格相对高低、波动扩张/收缩，以及突破或回归形态。",
        "",
        "### 扩展指标：ATR",
        "",
        "典型技术指标还包括 KDJ/随机指标、CCI、OBV、ADX、MFI、ROC、Williams %R、移动平均线系统等。本文选取 ATR 作为扩展指标，因为它能衡量波动幅度并考虑跳空。",
        "",
        "ATR 计算：",
        "",
        "- `TR = max(high-low, abs(high-prev_close), abs(low-prev_close))`",
        "- `ATR = WilderAverage(TR, 14)`",
        "- `ATR% = ATR / close * 100`",
        "",
        "常见作用：衡量波动率、辅助设置止损距离、比较不同阶段波动强弱；ATR 不判断方向。",
        "",
        "## 3. 最新指标快照",
        "",
        "| 数据集 | 日期 | 收盘价 | RSI14 | DIF | DEA | MACD柱 | BB上轨 | BB中轨 | BB下轨 | ATR14 | ATR% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in latest_rows:
        lines.append(
            f"| {row['label']} | {row['date']} | {row['close']:.2f} | {row['rsi_14']:.2f} | {row['macd_dif']:.4f} | {row['macd_dea']:.4f} | {row['macd_hist']:.4f} | {row['bb_upper_20_2']:.2f} | {row['bb_mid_20']:.2f} | {row['bb_lower_20_2']:.2f} | {row['atr_14']:.2f} | {row['atr_pct_14']:.2f}% |"
        )
    lines += [
        "",
        "## 4. 生成文件",
        "",
        "指标明细：",
        "",
    ]
    for label, path in indicator_paths.items():
        lines.append(f"- {label}: `{path}`")
    lines += ["", "图表：", ""]
    for label, path in chart_paths.items():
        lines.append(f"- {label}: `{path}`")
    lines += [
        "",
        "## 5. 参考来源",
        "",
    ]
    for url in source_urls:
        lines.append(f"- {url}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    chart_dir = output_dir / "charts"
    data_dir = output_dir / "data"
    inputs = [parse_input_spec(item) for item in args.input] if args.input else DEFAULT_INPUTS

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    desc_paths = {}
    indicator_paths = {}
    chart_paths = {}
    latest_rows = []

    for meta in inputs:
        raw, encoding = read_csv(meta["path"])
        meta["encoding"] = encoding
        df = prepare_data(raw)
        diag, desc = diagnose(df, meta)
        diagnostics.append(diag)

        slug = safe_slug(meta["label"])
        desc_path = data_dir / f"{slug}_descriptive_stats.csv"
        desc.to_csv(desc_path, encoding="utf-8-sig")
        desc_paths[meta["label"]] = str(desc_path)

        with_indicators = add_indicators(df)
        indicator_path = data_dir / f"{slug}_indicators.csv"
        with_indicators.to_csv(indicator_path, index=False, encoding="utf-8-sig")
        indicator_paths[meta["label"]] = str(indicator_path)

        chart_path = chart_dir / f"{slug}_technical_indicators.png"
        make_chart(with_indicators, meta["label"], chart_path)
        chart_paths[meta["label"]] = str(chart_path)

        latest = with_indicators.dropna(subset=["rsi_14", "bb_mid_20", "atr_14"]).iloc[-1]
        latest_rows.append(
            {
                "label": meta["label"],
                "date": latest["date"].date().isoformat(),
                "close": float(latest["close"]),
                "rsi_14": float(latest["rsi_14"]),
                "macd_dif": float(latest["macd_dif"]),
                "macd_dea": float(latest["macd_dea"]),
                "macd_hist": float(latest["macd_hist"]),
                "bb_upper_20_2": float(latest["bb_upper_20_2"]),
                "bb_mid_20": float(latest["bb_mid_20"]),
                "bb_lower_20_2": float(latest["bb_lower_20_2"]),
                "atr_14": float(latest["atr_14"]),
                "atr_pct_14": float(latest["atr_pct_14"]),
            }
        )

    diagnostics_path = data_dir / "data_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = data_dir / "latest_indicator_snapshot.csv"
    pd.DataFrame(latest_rows).to_csv(latest_path, index=False, encoding="utf-8-sig")

    report_path = output_dir / "stock_indicator_analysis_report.md"
    write_report(diagnostics, desc_paths, indicator_paths, chart_paths, latest_rows, report_path)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "diagnostics": str(diagnostics_path),
                "latest_snapshot": str(latest_path),
                "charts": chart_paths,
                "indicator_data": indicator_paths,
                "latest_rows": latest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
