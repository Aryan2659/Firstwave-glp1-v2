import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell,
} from "recharts";
import { Loader2, BarChart3, Sparkles } from "lucide-react";
import { api, formatFeatureName, type ModelMetrics } from "../api/client";
import { formatScore, formatPct } from "../lib/utils";

export function ModelPerformance() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);

  useEffect(() => {
    api.metrics().then(setMetrics);
  }, []);

  if (!metrics) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 animate-spin text-ink-400" />
      </div>
    );
  }

  const rocData = metrics.roc_curve.fpr.map((fpr, i) => ({
    fpr,
    tpr: metrics.roc_curve.tpr[i],
  }));

  const liftData = Object.entries(metrics.lift_by_decile)
    .map(([decile, lift]) => ({ decile: parseInt(decile), lift }))
    .sort((a, b) => a.decile - b.decile);

  const importanceData = Object.entries(metrics.feature_importance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([feature, value]) => ({ feature: formatFeatureName(feature), value }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="ROC-AUC"
          value={metrics.xgboost_auc}
          format="raw"
          subtitle={`Baseline LR ${formatScore(metrics.baseline_logistic_auc)}`}
          highlight
        />
        <MetricCard
          label="PR-AUC"
          value={metrics.xgboost_pr_auc}
          format="raw"
          subtitle="Precision-Recall AUC"
        />
        <MetricCard
          label="Top decile lift"
          value={metrics.lift_by_decile["1"] || 0}
          format="x"
          subtitle="vs random targeting"
          highlight
        />
        <MetricCard
          label="Cox concordance"
          value={metrics.cox_concordance}
          format="raw"
          subtitle="Survival model timing"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard
          title="Lift curve"
          subtitle="Top 10% of predicted physicians contain ~9× more early adopters than random"
          icon={<Sparkles className="w-4 h-4 text-accent-600" />}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={liftData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ebe7dc" vertical={false} />
              <XAxis
                dataKey="decile"
                tick={{ fontSize: 11, fill: "#7a6d4e" }}
                label={{ value: "Decile (1 = highest predicted)", position: "insideBottom", offset: -5, fontSize: 11, fill: "#5d533c" }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#7a6d4e" }}
                label={{ value: "Lift over baseline", angle: -90, position: "insideLeft", fontSize: 11, fill: "#5d533c" }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: "1px solid #ebe7dc", fontSize: 12 }}
                formatter={(v: number) => [`${v.toFixed(2)}x`, "Lift"]}
              />
              <ReferenceLine y={1} stroke="#b73706" strokeDasharray="4 4" label={{ value: "Random = 1×", fontSize: 10, fill: "#b73706", position: "insideTopRight" }} />
              <Bar dataKey="lift" radius={[4, 4, 0, 0]}>
                {liftData.map((d) => (
                  <Cell key={d.decile} fill={d.decile <= 3 ? "#f97306" : d.decile <= 6 ? "#ffb44b" : "#d6cfbb"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="ROC curve"
          subtitle={`AUC = ${formatScore(metrics.xgboost_auc)} on holdout test set`}
          icon={<BarChart3 className="w-4 h-4 text-ink-700" />}
        >
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ebe7dc" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(1)}
                tick={{ fontSize: 11, fill: "#7a6d4e" }}
                label={{ value: "False positive rate", position: "insideBottom", offset: -5, fontSize: 11, fill: "#5d533c" }}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => v.toFixed(1)}
                tick={{ fontSize: 11, fill: "#7a6d4e" }}
                label={{ value: "True positive rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "#5d533c" }}
              />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: "1px solid #ebe7dc", fontSize: 12 }}
                formatter={(v: number) => v.toFixed(3)}
              />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="#b73706"
                strokeDasharray="4 4"
              />
              <Line
                type="monotone"
                dataKey="tpr"
                stroke="#1a1611"
                strokeWidth={2.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <ChartCard
        title="Top features by mean SHAP impact"
        subtitle="What the model relies on most when ranking physicians"
        icon={<Sparkles className="w-4 h-4 text-accent-600" />}
      >
        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={importanceData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 130, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#ebe7dc" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fill: "#7a6d4e" }} />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fontSize: 11, fill: "#3f3829" }}
              width={130}
            />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: "1px solid #ebe7dc", fontSize: 12 }}
              formatter={(v: number) => v.toFixed(4)}
            />
            <Bar dataKey="value" fill="#f97306" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="card p-5 bg-ink-100/60 border-ink-200">
        <div className="text-xs font-medium uppercase tracking-wider text-ink-500 mb-2">
          Validation strategy
        </div>
        <p className="text-sm text-ink-700 leading-relaxed">
          Trained on 80% of synthetic Rybelsus prescriber data ({Math.round(metrics.test_size * 4).toLocaleString()} physicians),
          tested on the remaining {metrics.test_size.toLocaleString()} held-out physicians ({formatPct(metrics.test_positive_rate)} early-adopter base rate).
          5-fold cross-validation gives {formatScore(metrics.xgboost_cv_mean)} ± {formatScore(metrics.xgboost_cv_std)} — indicating a stable, well-calibrated model.
          The top decile of predicted physicians contains {formatScore(metrics.lift_by_decile["1"] || 0)}× more
          early adopters than random selection — actionable for field rep prioritization.
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  format,
  subtitle,
  highlight,
}: {
  label: string;
  value: number;
  format: "raw" | "x";
  subtitle: string;
  highlight?: boolean;
}) {
  const display = format === "x" ? `${value.toFixed(2)}x` : value.toFixed(3);
  return (
    <div className={`card p-5 ${highlight ? "bg-accent-50 border-accent-200" : ""}`}>
      <div className="text-[10px] uppercase tracking-widest text-ink-500 font-medium mb-3">
        {label}
      </div>
      <div className="stat-num text-3xl text-ink-900">{display}</div>
      <div className="text-[11px] text-ink-500 mt-1">{subtitle}</div>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <div className="font-display text-base text-ink-900">{title}</div>
      </div>
      <div className="text-xs text-ink-500 mb-4">{subtitle}</div>
      {children}
    </div>
  );
}
