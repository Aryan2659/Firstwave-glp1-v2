import { describe, it, expect } from "vitest";
import { formatScore, formatDays, formatUsd, formatPct, tierColor } from "../lib/utils";
import { formatFeatureName } from "../api/client";

describe("formatScore", () => {
  it("formats to 3 decimal places", () => {
    expect(formatScore(0.123456)).toBe("0.123");
    expect(formatScore(1.0)).toBe("1.000");
    expect(formatScore(0)).toBe("0.000");
  });
});

describe("formatDays", () => {
  it("handles sub-day", () => {
    expect(formatDays(0.5)).toBe("<1 day");
  });
  it("formats days", () => {
    expect(formatDays(15)).toBe("15 days");
  });
  it("converts to months past 60 days", () => {
    expect(formatDays(90)).toBe("3 mo");
  });
});

describe("formatUsd", () => {
  it("handles zero", () => {
    expect(formatUsd(0)).toBe("$0");
  });
  it("formats raw amount under 1000", () => {
    expect(formatUsd(450)).toBe("$450");
  });
  it("formats k for thousands", () => {
    expect(formatUsd(5500)).toBe("$5.5k");
  });
  it("formats M for millions", () => {
    expect(formatUsd(2_500_000)).toBe("$2.5M");
  });
});

describe("formatPct", () => {
  it("formats percentages", () => {
    expect(formatPct(0.05)).toBe("5.0%");
    expect(formatPct(0.123)).toBe("12.3%");
  });
});

describe("tierColor", () => {
  it("returns distinctive class for tier 1", () => {
    expect(tierColor("Tier 1")).toContain("accent");
  });
  it("returns muted class for tier 4", () => {
    expect(tierColor("Tier 4")).toContain("ink");
  });
});

describe("formatFeatureName", () => {
  it("strips one-hot prefixes", () => {
    expect(formatFeatureName("specialty_Endocrinology")).toBe("Endocrinology");
    expect(formatFeatureName("state_CA")).toBe("CA");
  });
  it("humanizes feature names", () => {
    expect(formatFeatureName("kol_pagerank_normalized")).toBe("KOL influence (PageRank)");
    expect(formatFeatureName("prior_injectable_glp1_prescriber")).toBe("Prior injectable GLP-1");
  });
});
