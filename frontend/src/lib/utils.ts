import clsx from "clsx";

export function cn(...inputs: Parameters<typeof clsx>) {
  return clsx(...inputs);
}

export function formatPct(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function formatScore(n: number): string {
  return n.toFixed(3);
}

export function formatDays(n: number): string {
  if (n < 1) return "<1 day";
  if (n < 60) return `${Math.round(n)} days`;
  return `${Math.round(n / 30)} mo`;
}

export function formatUsd(n: number): string {
  if (n === 0) return "$0";
  if (n < 1000) return `$${n.toFixed(0)}`;
  if (n < 1_000_000) return `$${(n / 1000).toFixed(1)}k`;
  return `$${(n / 1_000_000).toFixed(1)}M`;
}

export function tierColor(tier: string): string {
  switch (tier) {
    case "Tier 1":
      return "bg-accent-100 text-accent-800 border-accent-200";
    case "Tier 2":
      return "bg-amber-50 text-amber-800 border-amber-100";
    case "Tier 3":
      return "bg-ink-100 text-ink-700 border-ink-200";
    default:
      return "bg-ink-50 text-ink-500 border-ink-100";
  }
}
