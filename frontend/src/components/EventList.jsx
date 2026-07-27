import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import { severityTone, StatusBadge } from "./StatusBadge";

export default function EventList({ events, compact = false }) {
  return (
    <div className={compact ? "event-list compact" : "event-list"}>
      {events.map((event) => (
        <article className="event-item" key={event.id}>
          <div className={`event-icon ${severityTone(event.severity)}`}>
            {event.severity === "Critical" ? (
              <ShieldAlert size={18} />
            ) : event.severity === "Normal" ? (
              <CheckCircle2 size={18} />
            ) : (
              <AlertTriangle size={18} />
            )}
          </div>
          <div>
            <div className="event-title">
              <strong>{event.type}</strong>
              <StatusBadge value={event.severity} tone={severityTone(event.severity)} />
            </div>
            <p>{eventText(event)}</p>
            <div className="event-meta">
              <span>{event.time}</span>
              <span>{event.person}</span>
              <span>{event.location}</span>
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

function eventText(event) {
  const value = event.message ?? event.details;
  if (value == null) return "No additional details.";
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.map(String).join(", ");
  if (typeof value === "object") {
    if (typeof value.message === "string") return value.message;
    try {
      return JSON.stringify(value);
    } catch {
      return "Structured event details available.";
    }
  }
  return String(value);
}
