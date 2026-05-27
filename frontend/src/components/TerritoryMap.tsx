import { useEffect, useState } from "react";
import { Loader2, Map } from "lucide-react";
import { api, type Territory } from "../api/client";
import { formatScore } from "../lib/utils";

export function TerritoryMap() {
  const [territories, setTerritories] = useState<Territory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.territory().then((data) => {
      setTerritories(data.territories);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="w-6 h-6 animate-spin text-ink-400" />
      </div>
    );
  }

  const maxScore = Math.max(...territories.map((t) => t.mean_adoption_score));
  const maxTier1 = Math.max(...territories.map((t) => t.tier_1_count));

  return (
    <div className="space-y-6">
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-1">
          <Map className="w-4 h-4 text-ink-500" />
          <div className="text-[10px] uppercase tracking-widest text-ink-500 font-medium">
            State-level adoption potential
          </div>
        </div>
        <div className="font-display text-2xl text-ink-900 mb-1">
          {territories.length} territories ranked by mean adoption score
        </div>
        <div className="text-sm text-ink-500">
          Aggregated across all physicians per state. Deeper amber = higher launch readiness.
        </div>
      </div>

      <div className="card p-5">
        <table className="w-full">
          <thead>
            <tr className="text-left text-xs font-medium text-ink-500 uppercase tracking-wider border-b border-ink-100">
              <th className="py-3 px-3">Rank</th>
              <th className="py-3 px-3">State</th>
              <th className="py-3 px-3 text-right">Physicians</th>
              <th className="py-3 px-3">Mean adoption potential</th>
              <th className="py-3 px-3 text-right">High potential</th>
              <th className="py-3 px-3 text-right">Tier 1</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-50">
            {territories.map((t, idx) => {
              const widthPct = (t.mean_adoption_score / maxScore) * 100;
              const tier1Pct = maxTier1 ? (t.tier_1_count / maxTier1) * 100 : 0;
              return (
                <tr key={t.state}>
                  <td className="py-3 px-3 font-display text-ink-400 text-sm">
                    {String(idx + 1).padStart(2, "0")}
                  </td>
                  <td className="py-3 px-3 font-medium text-ink-900">{t.state}</td>
                  <td className="py-3 px-3 text-right text-sm text-ink-700">
                    {t.physician_count.toLocaleString()}
                  </td>
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-2 bg-ink-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-accent-300 to-accent-500"
                          style={{ width: `${widthPct}%` }}
                        />
                      </div>
                      <span className="stat-num text-sm text-ink-900 w-12 text-right">
                        {formatScore(t.mean_adoption_score)}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-3 text-right text-sm text-ink-700">
                    {t.high_potential_count}
                  </td>
                  <td className="py-3 px-3 text-right">
                    <div className="inline-flex items-center gap-2">
                      <div className="w-8 h-1.5 bg-ink-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-500"
                          style={{ width: `${tier1Pct}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-accent-700 w-6 text-right">
                        {t.tier_1_count}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
