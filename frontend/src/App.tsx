import { useEffect, useState } from "react";
import { Header } from "./components/Header";
import { PhysicianTable } from "./components/PhysicianTable";
import { PhysicianDetailDrawer } from "./components/PhysicianDetailDrawer";
import { TerritoryMap } from "./components/TerritoryMap";
import { ModelPerformance } from "./components/ModelPerformance";
import { api } from "./api/client";

type Tab = "rank" | "territory" | "model";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("rank");
  const [states, setStates] = useState<string[]>([]);
  const [specialties, setSpecialties] = useState<string[]>([]);
  const [selectedNpi, setSelectedNpi] = useState<string | null>(null);

  useEffect(() => {
    api.filters().then((data) => {
      setStates(data.states);
      setSpecialties(data.specialties);
    });
  }, []);

  return (
    <div className="min-h-screen bg-ink-50">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === "rank" && (
          <>
            <div className="mb-6 max-w-2xl">
              <h1 className="font-display text-3xl text-ink-900 leading-tight mb-2">
                Which physicians will prescribe oral GLP-1 first?
              </h1>
              <p className="text-ink-600 leading-relaxed">
                A ranked, explainable list of physicians most likely to be early adopters —
                with predicted timing and SHAP-driven rationale. Built for sales reps prioritizing
                Foundayo launch outreach.
              </p>
            </div>
            <PhysicianTable
              states={states}
              specialties={specialties}
              onSelectPhysician={setSelectedNpi}
            />
          </>
        )}

        {activeTab === "territory" && (
          <>
            <div className="mb-6 max-w-2xl">
              <h1 className="font-display text-3xl text-ink-900 leading-tight mb-2">
                Where is the launch readiest?
              </h1>
              <p className="text-ink-600 leading-relaxed">
                Mean adoption potential by state, ranked by aggregated physician scores.
                Higher amber bars = territories where launch resources should deploy first.
              </p>
            </div>
            <TerritoryMap />
          </>
        )}

        {activeTab === "model" && (
          <>
            <div className="mb-6 max-w-2xl">
              <h1 className="font-display text-3xl text-ink-900 leading-tight mb-2">
                How well does the model perform?
              </h1>
              <p className="text-ink-600 leading-relaxed">
                XGBoost classifier on Rybelsus historical adoption data, with Cox PH survival for timing.
                Trained pre-launch, validated on held-out physicians.
              </p>
            </div>
            <ModelPerformance />
          </>
        )}
      </main>

      {selectedNpi && (
        <PhysicianDetailDrawer
          npi={selectedNpi}
          onClose={() => setSelectedNpi(null)}
        />
      )}

      <footer className="border-t border-ink-100 mt-16 py-6">
        <div className="max-w-7xl mx-auto px-6 text-xs text-ink-500 flex items-center justify-between">
          <span>FirstWave v1.0 · Built on synthetic prescriber data modeled after CMS sources</span>
          <span className="mono">XGBoost + Cox PH + SHAP · FastAPI · React</span>
        </div>
      </footer>
    </div>
  );
}
