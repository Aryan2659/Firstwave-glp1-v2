import { useEffect, useState } from "react";
import { ChevronRight, Loader2, TrendingUp, Award } from "lucide-react";
import { api, type Physician } from "../api/client";
import { cn, formatDays, formatScore, tierColor } from "../lib/utils";

interface PhysicianTableProps {
  states: string[];
  specialties: string[];
  onSelectPhysician: (npi: string) => void;
}

export function PhysicianTable({ states, specialties, onSelectPhysician }: PhysicianTableProps) {
  const [physicians, setPhysicians] = useState<Physician[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [state, setState] = useState<string>("ALL");
  const [specialty, setSpecialty] = useState<string>("ALL");
  const [minScore, setMinScore] = useState(0);

  useEffect(() => {
    setLoading(true);
    api.predict({ state, specialty, min_score: minScore, limit: 50 })
      .then((data) => {
        setPhysicians(data.physicians);
        setTotal(data.total);
      })
      .finally(() => setLoading(false));
  }, [state, specialty, minScore]);

  const tier1Count = physicians.filter((p) => p.urgency_tier === "Tier 1").length;
  const meanScore = physicians.length
    ? physicians.reduce((a, p) => a + p.adoption_score, 0) / physicians.length
    : 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={<TrendingUp className="w-4 h-4" />}
          label="Top physicians shown"
          value={`${physicians.length}`}
          subtitle={`of ${total.toLocaleString()} matching filters`}
        />
        <StatCard
          icon={<Award className="w-4 h-4" />}
          label="Mean adoption score"
          value={formatScore(meanScore)}
          subtitle="among visible results"
        />
        <StatCard
          icon={<TrendingUp className="w-4 h-4 text-accent-600" />}
          label="Tier 1 candidates"
          value={`${tier1Count}`}
          subtitle="ready within 30 days"
          highlight
        />
      </div>

      <div className="card p-5">
        <div className="flex flex-wrap items-end gap-4 mb-5">
          <FilterSelect
            label="State"
            value={state}
            onChange={setState}
            options={[{ value: "ALL", label: "All states" }, ...states.map((s) => ({ value: s, label: s }))]}
          />
          <FilterSelect
            label="Specialty"
            value={specialty}
            onChange={setSpecialty}
            options={[{ value: "ALL", label: "All specialties" }, ...specialties.map((s) => ({ value: s, label: s }))]}
          />
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-ink-500 uppercase tracking-wider">
              Min score: {formatScore(minScore)}
            </label>
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={minScore}
              onChange={(e) => setMinScore(parseFloat(e.target.value))}
              className="w-48 accent-accent-500"
            />
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12 text-ink-500">
            <Loader2 className="w-5 h-5 animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto -mx-5">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs font-medium text-ink-500 uppercase tracking-wider border-b border-ink-100">
                  <th className="px-5 py-3">Rank</th>
                  <th className="px-3 py-3">NPI</th>
                  <th className="px-3 py-3">Specialty</th>
                  <th className="px-3 py-3">State</th>
                  <th className="px-3 py-3 text-right">Score</th>
                  <th className="px-3 py-3">Urgency</th>
                  <th className="px-3 py-3 text-right">When</th>
                  <th className="px-3 py-3 text-right">KOL</th>
                  <th className="px-5 py-3"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-50">
                {physicians.map((p, idx) => (
                  <tr
                    key={p.NPI}
                    onClick={() => onSelectPhysician(p.NPI)}
                    className="cursor-pointer hover:bg-ink-50 transition group"
                  >
                    <td className="px-5 py-3 font-display text-ink-400 text-sm">
                      {String(idx + 1).padStart(2, "0")}
                    </td>
                    <td className="px-3 py-3 mono text-xs text-ink-600">{p.NPI}</td>
                    <td className="px-3 py-3 text-sm text-ink-800">{p.specialty}</td>
                    <td className="px-3 py-3 text-sm text-ink-600">{p.state}</td>
                    <td className="px-3 py-3 text-right">
                      <span className="stat-num text-base text-ink-900">
                        {formatScore(p.adoption_score)}
                      </span>
                    </td>
                    <td className="px-3 py-3">
                      <span className={cn("badge border", tierColor(p.urgency_tier))}>
                        {p.urgency_tier}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-right text-sm text-ink-600">
                      {formatDays(p.predicted_days_to_rx)}
                    </td>
                    <td className="px-3 py-3 text-right text-sm">
                      {p.kol_pagerank_normalized > 0 ? (
                        <span className="text-accent-700 font-medium">
                          {(p.kol_pagerank_normalized * 100).toFixed(0)}
                        </span>
                      ) : (
                        <span className="text-ink-300">—</span>
                      )}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <ChevronRight className="w-4 h-4 text-ink-300 group-hover:text-accent-500 transition" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  icon, label, value, subtitle, highlight,
}: {
  icon: React.ReactNode; label: string; value: string; subtitle: string; highlight?: boolean;
}) {
  return (
    <div className={cn("card p-5", highlight && "bg-accent-50 border-accent-200")}>
      <div className="flex items-center gap-2 text-ink-500 mb-3">
        {icon}
        <span className="text-[11px] uppercase tracking-wider font-medium">{label}</span>
      </div>
      <div className="stat-num text-3xl text-ink-900">{value}</div>
      <div className="text-xs text-ink-500 mt-1">{subtitle}</div>
    </div>
  );
}

function FilterSelect({
  label, value, onChange, options,
}: {
  label: string; value: string; onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-medium text-ink-500 uppercase tracking-wider">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="px-3 py-1.5 rounded-lg border border-ink-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-accent-300"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}
