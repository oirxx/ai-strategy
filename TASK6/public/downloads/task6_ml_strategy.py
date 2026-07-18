"""TASK6：机器学习季度选股策略与样本外回测。

研究流程：
1. 读取股票财务特征与Tushare季度端点收盘价；
2. 构造下一季度收益和横截面高收益标签；
3. 使用扩展窗口训练逻辑回归、决策树、随机森林；
4. 每季度选择预测概率最高的20%股票并等权持有；
5. 输出分类指标、季度收益、净值、回撤、分组收益及图表。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FEATURE_PATH = DATA_DIR / "model_data_stock.csv"
PRICE_PATH = DATA_DIR / "quarterly_prices_tushare.csv"
RANDOM_STATE = 42
TOP_FRACTION = 0.20
ONE_WAY_COST = 0.001

MODEL_LABELS = {
    "logistic_regression": "逻辑回归",
    "decision_tree": "决策树",
    "random_forest": "随机森林",
}
MODEL_COLORS = {
    "logistic_regression": "#efb64f",
    "decision_tree": "#38bfc4",
    "random_forest": "#4d70f4",
    "benchmark": "#8291a5",
}
COLORS = {
    "navy": "#10263f",
    "blue": "#4d70f4",
    "teal": "#38bfc4",
    "gold": "#efb64f",
    "red": "#e76870",
    "ink": "#1b2a3a",
    "muted": "#718094",
    "grid": "#dce4ec",
}


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid")
    for font_path in [
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]:
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = fm.FontProperties(
                fname=font_path
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 240


def to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8", "9")):
        return f"{code}.BJ"
    return f"{code}.SH"


def load_and_merge_data() -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(FEATURE_PATH, dtype={"Code": str})
    features["Date"] = pd.to_datetime(features["Date"])
    features["Code"] = features["Code"].str.zfill(6)
    features["ts_code"] = features["Code"].map(to_ts_code)

    prices = pd.read_csv(PRICE_PATH, dtype={"ts_code": str, "trade_date": str})
    prices["Date"] = pd.to_datetime(prices["trade_date"])
    price_wide = prices.pivot_table(
        index="ts_code", columns="Date", values="price_value", aggfunc="last"
    )

    dates = sorted(features["Date"].unique())
    next_dates = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    price_dates = sorted(prices["Date"].unique())
    if len(price_dates) <= len(dates):
        raise ValueError("价格数据缺少最后一个持有期终点。")
    next_dates[dates[-1]] = price_dates[-1]

    frames: list[pd.DataFrame] = []
    for date in dates:
        frame = features.loc[features["Date"] == date].copy()
        next_date = pd.Timestamp(next_dates[date])
        current_price = price_wide.get(pd.Timestamp(date))
        next_price = price_wide.get(next_date)
        if current_price is None or next_price is None:
            raise ValueError(f"缺少价格端点：{date} -> {next_date}")
        frame["price"] = frame["ts_code"].map(current_price)
        frame["next_price"] = frame["ts_code"].map(next_price)
        frame["next_date"] = next_date
        frame["forward_return"] = (
            frame["next_price"] / frame["price"] - 1
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    raw_features = [
        column
        for column in features.columns
        if column not in {"Date", "Code", "ts_code", "Y"}
    ]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=raw_features + ["forward_return"]).copy()

    # 仅用于降低极少数公司行为或价格异常对组合均值的支配，保留每期99%样本区间。
    data["return_for_backtest"] = data.groupby("Date")["forward_return"].transform(
        lambda series: series.clip(
            series.quantile(0.005), series.quantile(0.995)
        )
    )
    median_return = data.groupby("Date")["forward_return"].transform("median")
    data["target"] = (data["forward_return"] > median_return).astype(int)
    data["benchmark_return"] = data.groupby("Date")[
        "return_for_backtest"
    ].transform("mean")
    return data, raw_features


def derive_features(
    data: pd.DataFrame, raw_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    engineered = data.copy()
    ranked_names: list[str] = []
    for column in raw_features:
        ranked = f"rank__{column}"
        engineered[ranked] = engineered.groupby("Date")[column].transform(
            lambda series: series.clip(
                series.quantile(0.01), series.quantile(0.99)
            ).rank(pct=True)
        )
        ranked_names.append(ranked)

    value_columns = [
        "企业倍数(EV除EBITDA)",
        "市净率PB(MRQ)",
        "市现率PCF(经营现金流TTM)",
        "市盈率PE(TTM)",
        "市盈率PE(TTM,扣除非经常性损益)",
        "市销率PS(TTM)",
    ]
    growth_columns = [
        "净利润同比增长率",
        "利润总额(同比增长率)",
        "基本每股收益(同比增长率)",
        "营业利润(同比增长率)",
        "营业总收入(同比增长率)",
    ]
    quality_columns = [
        "净资产同比增长率",
        "总资产同比增长率",
        "现金净流量同比增长率",
        "营业利润(同比增长率)",
    ]
    engineered["factor__value"] = 1 - engineered[
        [f"rank__{column}" for column in value_columns]
    ].mean(axis=1)
    engineered["factor__growth"] = engineered[
        [f"rank__{column}" for column in growth_columns]
    ].mean(axis=1)
    engineered["factor__quality"] = engineered[
        [f"rank__{column}" for column in quality_columns]
    ].mean(axis=1)
    engineered["factor__size"] = engineered["rank__MV"]
    model_features = ranked_names + [
        "factor__value",
        "factor__growth",
        "factor__quality",
        "factor__size",
    ]
    return engineered, model_features


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.4,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=80,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def calculate_turnover(
    current_codes: list[str], previous_codes: list[str] | None
) -> float:
    current = set(current_codes)
    if previous_codes is None:
        return 1.0
    previous = set(previous_codes)
    if not current or not previous:
        return 1.0
    weights: dict[str, tuple[float, float]] = {}
    for code in current | previous:
        weights[code] = (
            1 / len(previous) if code in previous else 0.0,
            1 / len(current) if code in current else 0.0,
        )
    return 0.5 * sum(abs(new - old) for old, new in weights.values())


def backtest(
    data: pd.DataFrame, model_features: list[str]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    dates = sorted(data["Date"].unique())
    test_dates = dates[-3:]
    predictions: list[pd.DataFrame] = []
    quarterly_rows: list[dict] = []
    group_rows: list[dict] = []
    previous_holdings: dict[str, list[str] | None] = {
        key: None for key in MODEL_LABELS
    }
    final_models: dict[str, object] = {}

    for test_date in test_dates:
        train = data.loc[data["Date"] < test_date].copy()
        test = data.loc[data["Date"] == test_date].copy()
        models = build_models()

        benchmark = float(test["return_for_backtest"].mean())
        quarterly_rows.append(
            {
                "date": pd.Timestamp(test_date),
                "model_key": "benchmark",
                "model": "全市场等权",
                "gross_return": benchmark,
                "turnover": 0.0,
                "cost": 0.0,
                "net_return": benchmark,
                "holding_count": int(len(test)),
            }
        )

        for model_key, model in models.items():
            model.fit(train[model_features], train["target"])
            probability = model.predict_proba(test[model_features])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            result = test[
                [
                    "Date",
                    "next_date",
                    "Code",
                    "ts_code",
                    "target",
                    "forward_return",
                    "return_for_backtest",
                ]
            ].copy()
            result["model_key"] = model_key
            result["probability"] = probability
            result["prediction"] = prediction
            result["probability_rank"] = result["probability"].rank(pct=True)
            predictions.append(result)

            selected = result.loc[
                result["probability_rank"] > 1 - TOP_FRACTION
            ].sort_values("probability", ascending=False)
            codes = selected["ts_code"].tolist()
            turnover = calculate_turnover(codes, previous_holdings[model_key])
            previous_holdings[model_key] = codes
            gross = float(selected["return_for_backtest"].mean())
            cost = ONE_WAY_COST * turnover
            quarterly_rows.append(
                {
                    "date": pd.Timestamp(test_date),
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "gross_return": gross,
                    "turnover": turnover,
                    "cost": cost,
                    "net_return": gross - cost,
                    "holding_count": int(len(selected)),
                }
            )

            result["group"] = pd.qcut(
                result["probability"].rank(method="first"),
                5,
                labels=["G1", "G2", "G3", "G4", "G5"],
            )
            for group, group_frame in result.groupby("group", observed=True):
                group_rows.append(
                    {
                        "date": pd.Timestamp(test_date),
                        "model_key": model_key,
                        "model": MODEL_LABELS[model_key],
                        "group": str(group),
                        "return": float(group_frame["return_for_backtest"].mean()),
                        "count": int(len(group_frame)),
                    }
                )
            final_models[model_key] = model

    prediction_df = pd.concat(predictions, ignore_index=True)
    quarterly = pd.DataFrame(quarterly_rows)
    group_returns = pd.DataFrame(group_rows)

    metric_rows: list[dict] = []
    roc_payload: dict[str, list[dict[str, float]]] = {}
    for model_key, model_frame in prediction_df.groupby("model_key"):
        y_true = model_frame["target"].to_numpy()
        probability = model_frame["probability"].to_numpy()
        prediction = model_frame["prediction"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
        fpr, tpr, _ = roc_curve(y_true, probability)
        metric_rows.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "auc": roc_auc_score(y_true, probability),
                "accuracy": accuracy_score(y_true, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
                "precision": precision_score(y_true, prediction),
                "recall": recall_score(y_true, prediction),
                "f1": f1_score(y_true, prediction),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        indices = np.unique(
            np.linspace(0, len(fpr) - 1, min(180, len(fpr))).astype(int)
        )
        roc_payload[model_key] = [
            {"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)}
            for i in indices
        ]
    model_metrics = pd.DataFrame(metric_rows).sort_values("auc", ascending=False)

    return (
        prediction_df,
        quarterly,
        group_returns,
        model_metrics,
        {"final_models": final_models, "roc": roc_payload, "test_dates": test_dates},
    )


def strategy_statistics(quarterly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model_key, frame in quarterly.groupby("model_key"):
        frame = frame.sort_values("date")
        returns = frame["net_return"].astype(float)
        nav = (1 + returns).cumprod()
        drawdown = nav / nav.cummax() - 1
        periods = len(returns)
        cumulative = float(nav.iloc[-1] - 1)
        annualized = float((1 + cumulative) ** (4 / periods) - 1)
        volatility = float(returns.std(ddof=1) * np.sqrt(4))
        sharpe = annualized / volatility if volatility > 0 else np.nan
        rows.append(
            {
                "model_key": model_key,
                "strategy": (
                    "全市场等权"
                    if model_key == "benchmark"
                    else MODEL_LABELS[model_key]
                ),
                "cumulative_return": cumulative,
                "annualized_return": annualized,
                "annualized_volatility": volatility,
                "sharpe": sharpe,
                "max_drawdown": float(drawdown.min()),
                "win_rate": float((returns > 0).mean()),
                "average_turnover": float(frame["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cumulative_return", ascending=False)


def feature_importance(
    context: dict[str, object], model_features: list[str]
) -> pd.DataFrame:
    model = context["final_models"]["random_forest"]
    labels = {
        "factor__value": "综合价值因子",
        "factor__growth": "综合成长因子",
        "factor__quality": "综合质量因子",
        "factor__size": "规模因子",
    }
    rows = []
    for feature, importance in zip(model_features, model.feature_importances_):
        display = labels.get(feature, feature.replace("rank__", ""))
        rows.append({"feature": display, "importance": float(importance)})
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_outputs(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    quarterly: pd.DataFrame,
    group_returns: pd.DataFrame,
    model_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    context: dict[str, object],
    model_features: list[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT_DIR / "sample_out_of_time_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quarterly.to_csv(
        OUTPUT_DIR / "quarterly_strategy_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_returns.to_csv(
        OUTPUT_DIR / "probability_group_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    strategy_metrics.to_csv(
        OUTPUT_DIR / "strategy_metrics.csv", index=False, encoding="utf-8-sig"
    )
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    holdings = (
        predictions.loc[predictions["probability_rank"] > 1 - TOP_FRACTION]
        .sort_values(["Date", "model_key", "probability"], ascending=[True, True, False])
        .copy()
    )
    holdings.to_csv(
        OUTPUT_DIR / "selected_holdings.csv", index=False, encoding="utf-8-sig"
    )

    label_summary = (
        data.groupby("Date")
        .agg(
            samples=("target", "size"),
            positive_rate=("target", "mean"),
            median_return=("forward_return", "median"),
            universe_return=("return_for_backtest", "mean"),
        )
        .reset_index()
    )
    label_summary.to_csv(
        OUTPUT_DIR / "label_summary.csv", index=False, encoding="utf-8-sig"
    )

    q_json = quarterly.copy()
    q_json["date"] = q_json["date"].dt.strftime("%Y-%m-%d")
    g_json = group_returns.copy()
    g_json["date"] = g_json["date"].dt.strftime("%Y-%m-%d")
    label_json = label_summary.copy()
    label_json["Date"] = label_json["Date"].dt.strftime("%Y-%m-%d")
    web_payload = {
        "metadata": {
            "sample_count": int(len(data)),
            "stock_count": int(data["Code"].nunique()),
            "feature_count": int(len(model_features)),
            "source_dates": [
                pd.Timestamp(date).strftime("%Y-%m-%d")
                for date in sorted(data["Date"].unique())
            ],
            "test_dates": [
                pd.Timestamp(date).strftime("%Y-%m-%d")
                for date in context["test_dates"]
            ],
            "top_fraction": TOP_FRACTION,
            "one_way_cost": ONE_WAY_COST,
            "return_source": "Tushare季度末收盘价（简单价格收益）",
            "target_definition": "下一季度收益高于当季横截面中位数",
        },
        "model_metrics": model_metrics.round(6).to_dict(orient="records"),
        "strategy_metrics": strategy_metrics.round(6).to_dict(orient="records"),
        "quarterly_returns": q_json.round(6).to_dict(orient="records"),
        "group_returns": g_json.round(6).to_dict(orient="records"),
        "feature_importance": importance.head(12).round(6).to_dict(orient="records"),
        "label_summary": label_json.round(6).to_dict(orient="records"),
        "roc": context["roc"],
        "top_holdings": (
            holdings.groupby(["Date", "model_key"], group_keys=False)
            .head(8)
            .assign(Date=lambda frame: frame["Date"].dt.strftime("%Y-%m-%d"))
            [
                [
                    "Date",
                    "model_key",
                    "Code",
                    "probability",
                    "forward_return",
                ]
            ]
            .round(6)
            .to_dict(orient="records")
        ),
    }
    (OUTPUT_DIR / "dashboard_data.json").write_text(
        json.dumps(web_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def finish_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_label_distribution(data: pd.DataFrame) -> None:
    summary = (
        data.groupby("Date")["target"]
        .agg(samples="size", positive="sum", positive_rate="mean")
        .reset_index()
    )
    summary["negative"] = summary["samples"] - summary["positive"]
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["negative"], color=COLORS["navy"], label="Y=0")
    ax.bar(
        x,
        summary["positive"],
        bottom=summary["negative"],
        color=COLORS["teal"],
        label="Y=1",
    )
    ax.set_xticks(x, summary["Date"].dt.strftime("%Y-%m"))
    ax.set_ylabel("样本数")
    ax.set_title("各季度模型标签分布", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="x", visible=False)
    finish_figure("figure_1_label_distribution.png")


def plot_auc(model_metrics: pd.DataFrame) -> None:
    display = model_metrics.sort_values("auc")
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = [MODEL_COLORS[key] for key in display["model_key"]]
    bars = ax.barh(display["model"], display["auc"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=10)
    ax.axvline(0.5, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax.set_xlim(0.45, max(0.75, display["auc"].max() + 0.06))
    ax.set_xlabel("样本外AUC")
    ax.set_title("三类模型样本外AUC比较", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="y", visible=False)
    finish_figure("figure_2_model_auc.png")


def plot_quarterly_returns(quarterly: pd.DataFrame) -> None:
    pivot = quarterly.pivot(index="date", columns="model_key", values="net_return")
    order = ["benchmark", "logistic_regression", "decision_tree", "random_forest"]
    labels = ["全市场等权", "逻辑回归", "决策树", "随机森林"]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(pivot))
    width = 0.19
    for index, (key, label) in enumerate(zip(order, labels)):
        ax.bar(
            x + (index - 1.5) * width,
            pivot[key] * 100,
            width,
            label=label,
            color=MODEL_COLORS[key],
        )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x, pivot.index.strftime("%Y-%m"))
    ax.set_ylabel("季度收益率（%）")
    ax.set_title("样本外季度策略收益", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.grid(axis="x", visible=False)
    finish_figure("figure_3_quarterly_returns.png")


def plot_nav_and_drawdown(quarterly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 0.75]},
    )
    for model_key, frame in quarterly.groupby("model_key"):
        frame = frame.sort_values("date")
        nav = (1 + frame["net_return"]).cumprod()
        drawdown = nav / nav.cummax() - 1
        label = (
            "全市场等权"
            if model_key == "benchmark"
            else MODEL_LABELS[model_key]
        )
        axes[0].plot(
            frame["date"],
            nav,
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS[model_key],
            label=label,
        )
        axes[1].plot(
            frame["date"],
            drawdown * 100,
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS[model_key],
        )
    axes[0].set_title("策略累计净值与回撤", loc="left", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("累计净值")
    axes[0].legend(frameon=False, ncol=4, fontsize=9)
    axes[1].set_ylabel("回撤（%）")
    axes[1].set_xlabel("信号形成季度")
    axes[1].axhline(0, color=COLORS["muted"], linewidth=0.7)
    finish_figure("figure_4_nav_drawdown.png")


def plot_feature_importance(importance: pd.DataFrame) -> None:
    display = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    bars = ax.barh(
        display["feature"],
        display["importance"],
        color=sns.color_palette("crest", n_colors=len(display)),
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_xlabel("特征重要性")
    ax.set_title("随机森林前10项重要特征", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="y", visible=False)
    finish_figure("figure_5_feature_importance.png")


def plot_group_returns(group_returns: pd.DataFrame) -> None:
    display = (
        group_returns.loc[group_returns["model_key"] == "random_forest"]
        .groupby("group", observed=True)["return"]
        .mean()
        .reindex(["G1", "G2", "G3", "G4", "G5"])
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars = ax.bar(
        display.index,
        display.values * 100,
        color=["#243c58", "#315f78", "#3d8d95", "#42b7b0", "#4d70f4"],
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_ylabel("平均季度收益率（%）")
    ax.set_xlabel("随机森林预测概率分组（G5最高）")
    ax.set_title("随机森林预测概率五分组收益", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", visible=False)
    finish_figure("figure_6_group_returns.png")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
    configure_plotting()
    data, raw_features = load_and_merge_data()
    data, model_features = derive_features(data, raw_features)
    predictions, quarterly, groups, model_metrics, context = backtest(
        data, model_features
    )
    strategy_metrics = strategy_statistics(quarterly)
    importance = feature_importance(context, model_features)
    save_outputs(
        data,
        predictions,
        quarterly,
        groups,
        model_metrics,
        strategy_metrics,
        importance,
        context,
        model_features,
    )
    plot_label_distribution(data)
    plot_auc(model_metrics)
    plot_quarterly_returns(quarterly)
    plot_nav_and_drawdown(quarterly)
    plot_feature_importance(importance)
    plot_group_returns(groups)

    print("\nModel metrics")
    print(model_metrics[["model", "auc", "f1"]].to_string(index=False))
    print("\nStrategy metrics")
    print(
        strategy_metrics[
            ["strategy", "cumulative_return", "sharpe", "max_drawdown"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
