import { useEffect, useState } from "react";
import { X, Loader2, Stethoscope, MapPin, Users, Award, Activity } from "lucide-react";
import { api, formatFeatureName, type ExplainResult, type PhysicianDetail } from "../api/client";
import { cn, formatScore, formatDays, formatUsd, tierColor } from "../lib/utils";

interface PhysicianDetailDrawerProps {
  npi: string;
  onClose: () => void;
}

export function PhysicianDetailDrawer({ npi, onClose }: PhysicianDetailDrawerProps) {
  const [detail, setDetail] = useState<PhysicianDetail | null>(null);
  const [explanation, setExplanation] = useState<ExplainResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.physician(npi), api.explain(npi)])
      .then(([d, e]) => {
        setDetail(d);
        setExplanation(e);
      })
      .finally(() => setLoading(false));
  }, [npi]);

  return (
    <div
      className="fixed inset-0 bg-ink-900/40 backdrop-blur-sm z-50 flex items-stretch justify-end"
      onClick={onClose}
    >
      <div
        className="bg-ink-50 w-full max-w-2xl h-full overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-ink-50/95 backdrop-blur border-b border-ink-100 px-6 py-4 flex items-center justify-between z-10">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-ink-500 font-medium">
              Physician profile
            </div>
            <div className="font-display text-xl text-ink-900">NPI {npi}</div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-ink-100 transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        {loading || !detail || !explanation ? (
          <div className="flex items-center justify-center py-32">
            <Loader2 className="w-6 h-6 animate-spin text-ink-400" />
          </div>
        ) : (
          <div className="px-6 py-6 space-y-6">
            <ScoreHero detail={detail} />
            <ProfileGrid detail={detail} />
            <ShapWaterfall explanation={explanation} />
            <NarrativeExplanation detail={detail} explanation={explanation} />
          </div>
        )}
      </div>
    </div>
  );
}

function ScoreHero({ detail }: { detail: PhysicianDetail }) {
  return (
    <div className="card p-6 bg-gradient-to-br from-ink-900 to-ink-800 text-ink-50 border-0">
      <div className="flex items-start justify-between gap-6">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-ink-300 font-medium mb-2">
            Adoption probability
          </div>
          <div className="font-display text-5xl text-accent-200 mb-2">
            {formatScore(detail.adoption_score)}
          </div>
          <div className="text-sm text-ink-300">
            {detail.specialty} · {detail.state} · {detail.years_in_practice} yrs in practice
          </div>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase tracking-widest text-ink-300 font-medium mb-2">
            Predicted timing
          </div>
          <div className="font-display text-2xl text-ink-50">
            {formatDays(detail.predicted_days_to_rx)}
          </div>
          <span className={cn("badge border mt-2 inline-flex", tierColor(detail.urgency_tier))}>
            {detail.urgency_tier_label}
          </span>
        </div>
      </div>
    </div>
  );
}

function ProfileGrid({ detail }: { detail: PhysicianDetail }) {
  const items = [
    { icon: <Stethoscope className="w-4 h-4" />, label: "Specialty", value: detail.specialty },
    { icon: <MapPin className="w-4 h-4" />, label: "Location", value: `${detail.state}${detail.rural ? " (rural)" : ""}` },
    { icon: <Users className="w-4 h-4" />, label: "Panel size", value: detail.patient_panel_size.toLocaleString() },
    { icon: <Award className="w-4 h-4" />, label: "Open Payments", value: formatUsd(detail.total_open_payments_usd) },
    {
      icon: <Activity className="w-4 h-4" />,
      label: "Prior GLP-1 (injectable)",
      value: detail.prior_injectable_glp1_prescriber ? "Yes" : "No",
    },
    {
      icon: <Activity className="w-4 h-4" />,
      label: "Prior SGLT2",
      value: detail.prior_sglt2_prescriber ? "Yes" : "No",
    },
  ];

  return (
    <div className="card p-5">
      <div className="text-xs font-medium uppercase tracking-wider text-ink-500 mb-3">
        Physician profile
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-3">
        {items.map((it) => (
          <div key={it.label} className="flex items-center gap-2.5">
            <div className="text-ink-400">{it.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="text-[10px] uppercase tracking-wider text-ink-400">{it.label}</div>
              <div className="text-sm text-ink-800 truncate">{it.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShapWaterfall({ explanation }: { explanation: ExplainResult }) {
  const contributions = explanation.contributions.slice(0, 10);
  const maxAbs = Math.max(...contributions.map((c) => c.abs_shap));

  return (
    <div className="card p-5">
      <div className="flex items-baseline justify-between mb-1">
        <div>
          <div className="font-display text-lg text-ink-900">Why this score</div>
          <div className="text-xs text-ink-500">
            SHAP values — what pushed the prediction up or down
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-wider text-ink-500">
          Logit {explanation.prediction_logit.toFixed(2)}
        </div>
      </div>

      <div className="mt-5 space-y-2">
        {contributions.map((c) => {
          const isPositive = c.shap_value >= 0;
          const widthPct = (c.abs_shap / maxAbs) * 100;
          return (
            <div key={c.feature} className="flex items-center gap-3">
              <div className="w-44 text-right text-xs text-ink-700 truncate">
                {formatFeatureName(c.feature)}
              </div>
              <div className="flex-1 relative h-6 flex items-center">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-ink-200" />
                {isPositive ? (
                  <div
                    className="absolute left-1/2 h-5 bg-accent-400 rounded-r flex items-center justify-end pr-1.5"
                    style={{ width: `${widthPct / 2}%` }}
                  >
                    <span className="text-[10px] font-medium text-accent-900">
                      +{c.shap_value.toFixed(2)}
                    </span>
                  </div>
                ) : (
                  <div
                    className="absolute right-1/2 h-5 bg-ink-300 rounded-l flex items-center justify-start pl-1.5"
                    style={{ width: `${widthPct / 2}%` }}
                  >
                    <span className="text-[10px] font-medium text-ink-700">
                      {c.shap_value.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
              <div className="w-12 text-right text-[11px] mono text-ink-500">
                {typeof c.value === "number" && c.value !== 0 && c.value !== 1
                  ? c.value > 100
                    ? c.value.toFixed(0)
                    : c.value.toFixed(2)
                  : c.value === 1
                  ? "✓"
                  : ""}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex items-center gap-4 text-[10px] text-ink-500">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2 bg-accent-400 rounded-sm" /> Pushes adoption up
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2 bg-ink-300 rounded-sm" /> Pushes adoption down
        </div>
      </div>
    </div>
  );
}

function NarrativeExplanation({
  detail,
  explanation,
}: {
  detail: PhysicianDetail;
  explanation: ExplainResult;
}) {
  const topPositive = explanation.contributions
    .filter((c) => c.shap_value > 0)
    .slice(0, 2)
    .map((c) => formatFeatureName(c.feature).toLowerCase());

  const score = detail.adoption_score;
  const verdict =
    score > 0.7
      ? "high-priority target"
      : score > 0.4
      ? "moderate-priority target"
      : "lower-priority target";

  return (
    <div className="card p-5 bg-ink-100/60 border-ink-200">
      <div className="font-display text-base text-ink-900 mb-2">In plain English</div>
      <p className="text-sm text-ink-700 leading-relaxed">
        This {detail.specialty.toLowerCase()} in {detail.state} is a{" "}
        <span className="font-medium text-ink-900">{verdict}</span> for the oral GLP-1 launch.
        The model's confidence is driven primarily by{" "}
        <span className="font-medium text-ink-900">{topPositive[0]}</span>
        {topPositive[1] && (
          <>
            {" "}and <span className="font-medium text-ink-900">{topPositive[1]}</span>
          </>
        )}
        . Predicted to write first prescription within{" "}
        <span className="font-medium text-ink-900">{formatDays(detail.predicted_days_to_rx)}</span>.
      </p>
    </div>
  );
}
