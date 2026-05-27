import { Waves } from "lucide-react";

type Tab = "rank" | "territory" | "model";

interface HeaderProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "rank", label: "Ranked physicians" },
    { id: "territory", label: "Territory map" },
    { id: "model", label: "Model performance" },
  ];

  return (
    <header className="border-b border-ink-100 bg-ink-50/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-8">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-ink-900 flex items-center justify-center">
            <Waves className="w-4 h-4 text-accent-300" strokeWidth={2.5} />
          </div>
          <div className="leading-tight">
            <div className="font-display text-lg font-medium text-ink-900">
              FirstWave
            </div>
            <div className="text-[10px] uppercase tracking-widest text-ink-500 font-medium">
              Oral GLP-1 launch intelligence
            </div>
          </div>
        </div>

        <nav className="flex gap-1 ml-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`px-4 py-2 text-sm rounded-lg transition ${
                activeTab === tab.id
                  ? "bg-ink-900 text-ink-50"
                  : "text-ink-600 hover:bg-ink-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
