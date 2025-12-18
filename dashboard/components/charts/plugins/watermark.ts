import type { TextWatermarkOptions } from "lightweight-charts";

export function cbiTextWatermarkOptions(opts?: {
  line1?: string;
  line2?: string;
}): TextWatermarkOptions {
  const line1 = opts?.line1 ?? "ZINC DIGITAL";
  const line2 = opts?.line2 ?? "CBI‑V15";

  return {
    visible: true,
    horzAlign: "center",
    vertAlign: "center",
    lines: [
      {
        text: line1,
        fontSize: 56,
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji",
        color: "rgba(255,255,255,0.06)",
        lineHeight: 60,
        fontStyle: "600",
      },
      {
        text: line2,
        fontSize: 14,
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji",
        color: "rgba(255,255,255,0.12)",
        lineHeight: 18,
        fontStyle: "500",
      },
    ],
  };
}

