"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ModelKey =
  | "logistic_regression"
  | "decision_tree"
  | "random_forest";

type DashboardData = {
  metadata: {
    sample_count: number;
    stock_count: number;
    feature_count: number;
    source_dates: string[];
    test_dates: string[];
    top_fraction: number;
    one_way_cost: number;
    return_source: string;
    target_definition: string;
  };
  model_metrics: Array<{
    model_key: ModelKey;
    model: string;
    auc: number;
    accuracy: number;
    balanced_accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  }>;
  strategy_metrics: Array<{
    model_key: ModelKey | "benchmark";
    strategy: string;
    cumulative_return: number;
    annualized_return: number;
    annualized_volatility: number;
    sharpe: number;
    max_drawdown: number;
    win_rate: number;
    average_turnover: number;
  }>;
  quarterly_returns: Array<{
    date: string;
    model_key: ModelKey | "benchmark";
    model: string;
    net_return: number;
    turnover: number;
    holding_count: number;
  }>;
  group_returns: Array<{
    date: string;
    model_key: ModelKey;
    model: string;
    group: string;
    return: number;
  }>;
  feature_importance: Array<{ feature: string; importance: number }>;
  label_summary: Array<{
    Date: string;
    samples: number;
    positive_rate: number;
    universe_return: number;
  }>;
  roc: Record<ModelKey, Array<{ fpr: number; tpr: number }>>;
  top_holdings: Array<{
    Date: string;
    model_key: ModelKey;
    Code: string;
    probability: number;
    forward_return: number;
  }>;
};

const MODEL_META: Record<
  ModelKey,
  { name: string; short: string; color: string; thesis: string }
> = {
  logistic_regression: {
    name: "逻辑回归",
    short: "LR",
    color: "#d8ff55",
    thesis: "线性概率基线，稳定、透明，本次样本外表现最佳。",
  },
  decision_tree: {
    name: "决策树",
    short: "DT",
    color: "#ffb36b",
    thesis: "规则可读，但单树容易受样本切分影响。",
  },
  random_forest: {
    name: "随机森林",
    short: "RF",
    color: "#69e6d0",
    thesis: "以集成降低方差，并输出非线性因子重要性。",
  },
};

const pct = (value: number, digits = 1) =>
  `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
const num = (value: number, digits = 2) => value.toFixed(digits);
const dateLabel = (value: string) => value.slice(0, 7).replace("-", ".");

function useCanvas(
  draw: (context: CanvasRenderingContext2D, width: number, height: number) => void,
  dependencies: unknown[],
) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      draw(context, rect.width, rect.height);
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);
  return ref;
}

function NetValueChart({
  data,
  selected,
}: {
  data: DashboardData;
  selected: ModelKey;
}) {
  const rows = data.quarterly_returns.filter(
    (item) => item.model_key === selected || item.model_key === "benchmark",
  );
  const chartRef = useCanvas(
    (ctx, width, height) => {
      const pad = { left: 48, right: 18, top: 24, bottom: 34 };
      const keys: Array<ModelKey | "benchmark"> = [selected, "benchmark"];
      const series = keys.map((key) => {
        let nav = 1;
        return rows
          .filter((row) => row.model_key === key)
          .sort((a, b) => a.date.localeCompare(b.date))
          .map((row, index) => {
            nav *= 1 + row.net_return;
            return { ...row, nav, index };
          });
      });
      const values = [1, ...series.flat().map((point) => point.nav)];
      const min = Math.min(...values) - 0.025;
      const max = Math.max(...values) + 0.025;
      const x = (index: number) =>
        pad.left +
        (index / 3) * (width - pad.left - pad.right);
      const y = (value: number) =>
        pad.top +
        ((max - value) / (max - min)) * (height - pad.top - pad.bottom);

      ctx.font = "11px var(--font-mono)";
      ctx.strokeStyle = "rgba(210, 235, 224, .12)";
      ctx.fillStyle = "rgba(222, 236, 229, .55)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i += 1) {
        const gy = pad.top + (i / 3) * (height - pad.top - pad.bottom);
        ctx.beginPath();
        ctx.moveTo(pad.left, gy);
        ctx.lineTo(width - pad.right, gy);
        ctx.stroke();
        const value = max - (i / 3) * (max - min);
        ctx.fillText(value.toFixed(2), 8, gy + 4);
      }
      const dates = data.metadata.test_dates;
      dates.forEach((date, index) => {
        ctx.fillText(dateLabel(date), x(index + 1) - 20, height - 10);
      });

      series.forEach((points, seriesIndex) => {
        const color =
          seriesIndex === 0 ? MODEL_META[selected].color : "#70817a";
        const all = [{ nav: 1, index: 0 }, ...points.map((p) => ({ nav: p.nav, index: p.index + 1 }))];
        ctx.strokeStyle = color;
        ctx.lineWidth = seriesIndex === 0 ? 3 : 2;
        ctx.setLineDash(seriesIndex === 0 ? [] : [5, 5]);
        ctx.beginPath();
        all.forEach((point, index) => {
          if (index === 0) ctx.moveTo(x(point.index), y(point.nav));
          else ctx.lineTo(x(point.index), y(point.nav));
        });
        ctx.stroke();
        ctx.setLineDash([]);
        all.slice(1).forEach((point) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x(point.index), y(point.nav), 4, 0, Math.PI * 2);
          ctx.fill();
        });
      });
    },
    [data, selected],
  );
  return <canvas className="chart-canvas" ref={chartRef} aria-label="累计净值图" />;
}

function RocChart({
  data,
  selected,
}: {
  data: DashboardData;
  selected: ModelKey;
}) {
  const ref = useCanvas(
    (ctx, width, height) => {
      const pad = 34;
      const x = (v: number) => pad + v * (width - pad * 1.5);
      const y = (v: number) => height - pad - v * (height - pad * 1.5);
      ctx.strokeStyle = "rgba(210, 235, 224, .12)";
      ctx.lineWidth = 1;
      [0, 0.5, 1].forEach((v) => {
        ctx.beginPath();
        ctx.moveTo(x(v), y(0));
        ctx.lineTo(x(v), y(1));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x(0), y(v));
        ctx.lineTo(x(1), y(v));
        ctx.stroke();
      });
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "#65746f";
      ctx.beginPath();
      ctx.moveTo(x(0), y(0));
      ctx.lineTo(x(1), y(1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = MODEL_META[selected].color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      data.roc[selected].forEach((point, index) => {
        if (index === 0) ctx.moveTo(x(point.fpr), y(point.tpr));
        else ctx.lineTo(x(point.fpr), y(point.tpr));
      });
      ctx.stroke();
      ctx.fillStyle = "rgba(226, 239, 233, .55)";
      ctx.font = "10px var(--font-mono)";
      ctx.fillText("0", x(0) - 4, y(0) + 17);
      ctx.fillText("FPR → 1", x(1) - 34, y(0) + 17);
      ctx.fillText("TPR", 4, y(1) + 4);
    },
    [data, selected],
  );
  return <canvas className="roc-canvas" ref={ref} aria-label="ROC曲线图" />;
}

export default function Dashboard({ data }: { data: DashboardData }) {
  const [selected, setSelected] = useState<ModelKey>("logistic_regression");
  const [selectedDate, setSelectedDate] = useState(
    data.metadata.test_dates.at(-1) ?? "",
  );
  const modelMetric = data.model_metrics.find(
    (item) => item.model_key === selected,
  )!;
  const strategyMetric = data.strategy_metrics.find(
    (item) => item.model_key === selected,
  )!;
  const benchmark = data.strategy_metrics.find(
    (item) => item.model_key === "benchmark",
  )!;
  const quarters = data.quarterly_returns
    .filter((item) => item.model_key === selected)
    .sort((a, b) => a.date.localeCompare(b.date));
  const holdings = data.top_holdings.filter(
    (item) => item.model_key === selected && item.Date === selectedDate,
  );
  const groupReturns = useMemo(
    () =>
      ["G1", "G2", "G3", "G4", "G5"].map((group) => ({
        group,
        value:
          data.group_returns
            .filter(
              (item) => item.model_key === selected && item.group === group,
            )
            .reduce((sum, item) => sum + item.return, 0) /
          data.metadata.test_dates.length,
      })),
    [data, selected],
  );
  const relative = strategyMetric.cumulative_return - benchmark.cumulative_return;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="回到顶部">
          <span className="brand-mark">Q/06</span>
          <span>QUANT RESEARCH DESK</span>
        </a>
        <nav aria-label="页面导航">
          <a href="#backtest">回测</a>
          <a href="#models">模型</a>
          <a href="#portfolio">组合</a>
          <a href="#method">方法</a>
        </nav>
        <a className="download-small" href="/downloads/姓名TASK6.pdf">
          PDF ↗
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <div className="eyebrow">
            <span className="status-dot" />
            TASK 06 · OUT-OF-TIME RESEARCH
          </div>
          <h1>
            预测只是开始，
            <br />
            <span>收益才是验收。</span>
          </h1>
          <p>
            以 20,733 条股票—季度样本为起点，把财务因子压缩成概率，
            再把概率转化为季度持仓。三类模型，同一套样本外规则，结果不回避熊市。
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#backtest">
              打开回测台 <span>↘</span>
            </a>
            <a className="button ghost" href="/downloads/task6_ml_strategy.py">
              下载 Python
            </a>
          </div>
        </div>
        <aside className="hero-terminal">
          <div className="terminal-head">
            <span>RESEARCH / LIVE</span>
            <span className="terminal-dots">● ● ●</span>
          </div>
          <div className="terminal-main">
            <span className="terminal-label">BEST OOT MODEL</span>
            <strong>LOGISTIC<br />REGRESSION</strong>
            <div className="terminal-number">
              <span>AUC</span>
              <b>0.632</b>
            </div>
          </div>
          <div className="terminal-footer">
            <span>REL. TO BENCHMARK</span>
            <b>{pct(relative)}</b>
          </div>
        </aside>
        <div className="tape" aria-label="研究概要">
          {[
            ["SAMPLES", data.metadata.sample_count.toLocaleString()],
            ["STOCKS", data.metadata.stock_count.toLocaleString()],
            ["FACTORS", String(data.metadata.feature_count)],
            ["TEST WINDOWS", String(data.metadata.test_dates.length)],
            ["REBAlANCE", "QUARTERLY"],
          ].map(([label, value]) => (
            <div key={label}>
              <span>{label}</span>
              <b>{value}</b>
            </div>
          ))}
        </div>
      </section>

      <section className="desk section-shell" id="backtest">
        <div className="section-heading">
          <div>
            <span className="section-index">01 / BACKTEST TAPE</span>
            <h2>从概率，到净值。</h2>
          </div>
          <p>
            选择模型，查看相同持仓规则下的样本外表现。组合每季选择预测概率最高的
            20%，等权持有，并扣除单边 0.10% 换手成本。
          </p>
        </div>

        <div className="model-tabs" role="tablist" aria-label="选择模型">
          {(Object.keys(MODEL_META) as ModelKey[]).map((key) => (
            <button
              type="button"
              role="tab"
              aria-selected={selected === key}
              className={selected === key ? "active" : ""}
              key={key}
              onClick={() => setSelected(key)}
              style={{ "--model-color": MODEL_META[key].color } as React.CSSProperties}
            >
              <span>{MODEL_META[key].short}</span>
              {MODEL_META[key].name}
            </button>
          ))}
        </div>

        <div className="backtest-grid">
          <article className="panel nav-panel">
            <div className="panel-head">
              <div>
                <span>CUMULATIVE NAV</span>
                <h3>{MODEL_META[selected].name} vs 全市场等权</h3>
              </div>
              <div className="legend">
                <span style={{ color: MODEL_META[selected].color }}>● 模型</span>
                <span>● 基准</span>
              </div>
            </div>
            <NetValueChart data={data} selected={selected} />
          </article>
          <article className="panel score-panel">
            <span className="panel-kicker">PERFORMANCE SNAPSHOT</span>
            <div className="score-main">
              <span>三季度累计</span>
              <strong className={strategyMetric.cumulative_return >= 0 ? "positive" : "negative"}>
                {pct(strategyMetric.cumulative_return)}
              </strong>
              <small>
                相对基准 <b>{pct(relative)}</b>
              </small>
            </div>
            <div className="score-list">
              <div><span>最大回撤</span><b>{pct(strategyMetric.max_drawdown)}</b></div>
              <div><span>年化波动</span><b>{pct(strategyMetric.annualized_volatility)}</b></div>
              <div><span>Sharpe</span><b>{num(strategyMetric.sharpe)}</b></div>
              <div><span>平均换手</span><b>{pct(strategyMetric.average_turnover)}</b></div>
            </div>
          </article>
        </div>

        <div className="quarter-strip">
          {quarters.map((quarter, index) => (
            <article key={quarter.date}>
              <span>Q{index + 1} · {dateLabel(quarter.date)}</span>
              <strong className={quarter.net_return >= 0 ? "positive" : "negative"}>
                {pct(quarter.net_return)}
              </strong>
              <div className="mini-track">
                <i
                  style={{
                    width: `${Math.min(100, Math.abs(quarter.net_return) * 900)}%`,
                    background:
                      quarter.net_return >= 0 ? "#d8ff55" : "#ff7f6f",
                  }}
                />
              </div>
              <small>{quarter.holding_count} 只持仓 · 换手 {pct(quarter.turnover, 0)}</small>
            </article>
          ))}
        </div>
        <div className="truth-note">
          <b>研究结论 /</b>
          本测试期市场整体下跌，所有模型绝对收益均为负；逻辑回归通过因子筛选将跌幅从
          19.8% 收窄至 3.0%。这是风险缓冲，不是“稳定盈利”的证明。
        </div>
      </section>

      <section className="model-lab" id="models">
        <div className="section-shell">
          <div className="section-heading light">
            <div>
              <span className="section-index">02 / MODEL LAB</span>
              <h2>准确率不是终点。</h2>
            </div>
            <p>
              AUC 衡量模型对高收益股票的排序能力；F1 同时考虑查准率与召回率。
              这里的所有指标均来自滚动样本外预测。
            </p>
          </div>
          <div className="lab-grid">
            <article className="roc-card">
              <div className="metric-orbit">
                <span>AUC</span>
                <strong>{modelMetric.auc.toFixed(3)}</strong>
                <small>0.500 = 随机</small>
              </div>
              <RocChart data={data} selected={selected} />
            </article>
            <article className="confusion-card">
              <div className="card-title">
                <span>CONFUSION MATRIX</span>
                <h3>{MODEL_META[selected].name}</h3>
              </div>
              <div className="confusion-grid">
                <div className="axis-y">实际</div>
                <div className="axis-x">预测</div>
                <div className="cell muted"><span>TN</span><b>{modelMetric.tn.toLocaleString()}</b></div>
                <div className="cell warn"><span>FP</span><b>{modelMetric.fp.toLocaleString()}</b></div>
                <div className="cell warn"><span>FN</span><b>{modelMetric.fn.toLocaleString()}</b></div>
                <div className="cell hot"><span>TP</span><b>{modelMetric.tp.toLocaleString()}</b></div>
              </div>
              <div className="metric-row">
                <div><span>PRECISION</span><b>{modelMetric.precision.toFixed(3)}</b></div>
                <div><span>RECALL</span><b>{modelMetric.recall.toFixed(3)}</b></div>
                <div><span>F1</span><b>{modelMetric.f1.toFixed(3)}</b></div>
              </div>
            </article>
            <article className="ranking-card">
              <div className="card-title">
                <span>MODEL LEADERBOARD</span>
                <h3>样本外排序能力</h3>
              </div>
              <div className="leaderboard">
                {[...data.model_metrics]
                  .sort((a, b) => b.auc - a.auc)
                  .map((item, index) => (
                    <button
                      type="button"
                      key={item.model_key}
                      onClick={() => setSelected(item.model_key)}
                      className={selected === item.model_key ? "selected" : ""}
                    >
                      <span className="rank">0{index + 1}</span>
                      <span className="leader-name">{item.model}</span>
                      <span className="auc-bar">
                        <i style={{ width: `${((item.auc - 0.5) / 0.16) * 100}%` }} />
                      </span>
                      <b>{item.auc.toFixed(3)}</b>
                    </button>
                  ))}
              </div>
              <p>{MODEL_META[selected].thesis}</p>
            </article>
          </div>
        </div>
      </section>

      <section className="portfolio section-shell" id="portfolio">
        <div className="section-heading">
          <div>
            <span className="section-index">03 / PORTFOLIO ENGINE</span>
            <h2>模型如何变成持仓？</h2>
          </div>
          <p>
            先在每个季度内衍生横截面排名因子，再用历史季度训练模型，按预测概率从高到低选取前
            20%，下一季度验证收益。
          </p>
        </div>
        <div className="pipeline">
          {[
            ["01", "RAW", "财务与估值数据", "17项原始指标"],
            ["02", "FACTOR", "截面排名与合成", "21项模型因子"],
            ["03", "MODEL", "扩展窗口训练", "LR · DT · RF"],
            ["04", "SIGNAL", "高收益概率", "Top 20%"],
            ["05", "TRADE", "季度等权调仓", "成本后收益"],
          ].map(([number, tag, title, detail]) => (
            <div key={number}>
              <span className="pipe-number">{number}</span>
              <span className="pipe-tag">{tag}</span>
              <b>{title}</b>
              <small>{detail}</small>
            </div>
          ))}
        </div>
        <div className="portfolio-grid">
          <article className="holdings-panel">
            <div className="panel-head">
              <div>
                <span>TOP PROBABILITY NAMES</span>
                <h3>{MODEL_META[selected].name} · 概率前八</h3>
              </div>
              <div className="date-tabs">
                {data.metadata.test_dates.map((date) => (
                  <button
                    type="button"
                    key={date}
                    className={selectedDate === date ? "active" : ""}
                    onClick={() => setSelectedDate(date)}
                  >
                    {dateLabel(date)}
                  </button>
                ))}
              </div>
            </div>
            <div className="table">
              <div className="table-row table-head">
                <span>RANK</span><span>CODE</span><span>PROB.</span><span>NEXT QTR</span>
              </div>
              {holdings.map((item, index) => (
                <div className="table-row" key={item.Code}>
                  <span>0{index + 1}</span>
                  <b>{item.Code}</b>
                  <span>{pct(item.probability, 1)}</span>
                  <strong className={item.forward_return >= 0 ? "positive" : "negative"}>
                    {pct(item.forward_return)}
                  </strong>
                </div>
              ))}
            </div>
          </article>
          <article className="quintile-panel">
            <div className="card-title">
              <span>PROBABILITY QUINTILES</span>
              <h3>预测分组平均收益</h3>
            </div>
            <div className="quintiles">
              {groupReturns.map((item, index) => {
                const min = Math.min(...groupReturns.map((row) => row.value));
                const max = Math.max(...groupReturns.map((row) => row.value));
                const height = 36 + ((item.value - min) / Math.max(max - min, 0.001)) * 110;
                return (
                  <div key={item.group}>
                    <span>{pct(item.value)}</span>
                    <i style={{ height, opacity: 0.45 + index * 0.12 }} />
                    <b>{item.group}</b>
                    <small>{index === 0 ? "LOW" : index === 4 ? "HIGH" : ""}</small>
                  </div>
                );
              })}
            </div>
            <p>若 G5 长期高于 G1，说明概率排序具有经济意义；短样本下仍需谨慎验证。</p>
          </article>
        </div>
      </section>

      <section className="signal-section">
        <div className="section-shell">
          <div className="section-heading light">
            <div>
              <span className="section-index">04 / SIGNAL MAP</span>
              <h2>随机森林看重什么？</h2>
            </div>
            <p>
              特征重要性反映模型在当前样本中的分裂贡献，并不等同于因果关系。
              排名因子将不同量纲统一到 0—1 区间。
            </p>
          </div>
          <div className="signal-map">
            {data.feature_importance.slice(0, 10).map((item, index) => (
              <div key={item.feature} className={index < 3 ? "major" : ""}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <b>{item.feature.replace("rank__", "").replace("factor__", "合成·")}</b>
                <i style={{ width: `${(item.importance / data.feature_importance[0].importance) * 100}%` }} />
                <strong>{item.importance.toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="method section-shell" id="method">
        <div className="section-heading">
          <div>
            <span className="section-index">05 / RESEARCH NOTES</span>
            <h2>把边界写在结果旁边。</h2>
          </div>
          <p>
            机器学习策略的价值是系统化提取弱信号，不是消除市场不确定性。
            公开假设、成本与样本边界，才让结果可以复核。
          </p>
        </div>
        <div className="notes-grid">
          <article className="definition-card">
            <span className="note-number">A</span>
            <h3>自变量因子</h3>
            <p>
              估值、成长、质量、规模等财务指标的季度截面排名，以及价值、成长、质量、规模四类合成因子。
            </p>
            <b>17 RAW → 21 MODEL FEATURES</b>
          </article>
          <article className="definition-card">
            <span className="note-number">B</span>
            <h3>预测应变量</h3>
            <p>
              下一季度简单价格收益高于当季横截面中位数记为 1，否则记为 0；模型输出属于类别 1 的概率。
            </p>
            <b>FORWARD RETURN &gt; MEDIAN</b>
          </article>
          <article className="definition-card">
            <span className="note-number">C</span>
            <h3>交易规则</h3>
            <p>
              每期仅用过去数据训练，选择概率最高的 20% 股票等权持有一个季度，并按换手扣除交易成本。
            </p>
            <b>EXPANDING WINDOW · NO LOOK-AHEAD</b>
          </article>
        </div>
        <div className="pros-cons">
          <article>
            <span className="positive">+ ADVANTAGES</span>
            <h3>更广的信号空间</h3>
            <p>能同时处理大量因子、非线性与交互关系；流程可复现，适合持续迭代。</p>
          </article>
          <article>
            <span className="negative">− LIMITATIONS</span>
            <h3>过拟合与漂移</h3>
            <p>历史规律可能失效；换手、冲击成本与公司行为会让实盘弱于回测。</p>
          </article>
          <article className="warning">
            <span>⚠ STUDY BOUNDARY</span>
            <h3>只有 3 个样本外季度</h3>
            <p>
              本结果用于展示完整研究链路，不足以证明长期有效。正式投资前应扩展时间、采用复权总收益并加入行业与风格约束。
            </p>
          </article>
        </div>
      </section>

      <section className="download-section">
        <div className="section-shell download-inner">
          <div>
            <span className="section-index">DELIVERABLES / TASK 06</span>
            <h2>研究可读，代码可跑，结果可核。</h2>
          </div>
          <div className="download-cards">
            <a href="/downloads/姓名TASK6.pdf">
              <span>01</span><b>课程报告 PDF</b><small>宋体五号 · 图表与解读</small><i>↓</i>
            </a>
            <a href="/downloads/task6_ml_strategy.py">
              <span>02</span><b>完整 Python</b><small>特征 · 模型 · 回测</small><i>↓</i>
            </a>
            <a href="/data/dashboard_data.json">
              <span>03</span><b>网页数据 JSON</b><small>结构化研究结果</small><i>↓</i>
            </a>
          </div>
        </div>
      </section>

      <footer>
        <div>
          <span className="brand-mark">Q/06</span>
          <p>机器学习交易策略 · 教学研究作品</p>
        </div>
        <p>DATA: TUSHARE QUARTER-END CLOSE · NOT INVESTMENT ADVICE</p>
        <a href="#top">BACK TO TOP ↑</a>
      </footer>
    </main>
  );
}
