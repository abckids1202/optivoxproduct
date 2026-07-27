import clsx from "../utils/clsx";

export function StatusBadge({ value, tone = "neutral" }) {
  return <span className={clsx("status-badge", `tone-${tone}`)}>{value}</span>;
}

export function severityTone(severity) {
  const normalized = String(severity || "").toLowerCase();
  if (normalized === "critical") return "danger";
  if (normalized === "warning" || normalized === "attention") return "warning";
  if (normalized === "normal" || normalized === "operational") return "success";
  return "neutral";
}
