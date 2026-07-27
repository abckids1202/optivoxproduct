import { AlertTriangle, Camera, Filter, ShieldAlert } from "lucide-react";
import EventList from "../components/EventList";
import StatCard from "../components/StatCard";

export default function Security({ state }) {
  const critical = state.events.filter((e) => e.severity === "Critical").length;
  const warning = state.events.filter((e) => e.severity === "Warning" || e.severity === "Attention").length;
  const unknown = state.events.filter((e) => e.type.includes("UNKNOWN")).length;
  const snapshots = state.events.filter((e) => e.severity !== "Normal").length;

  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Critical" value={critical} detail="Immediate attention" icon={ShieldAlert} tone="danger" />
        <StatCard label="Warnings" value={warning} detail="Needs review" icon={AlertTriangle} tone="warning" />
        <StatCard label="Unknown people" value={unknown} detail="Presence only" icon={AlertTriangle} tone="info" />
        <StatCard label="Snapshots" value={snapshots} detail="Evidence saved" icon={Camera} tone="neutral" />
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Security history</p>
            <h2>Event review</h2>
          </div>
          <button type="button">
            <Filter size={16} />
            Filter
          </button>
        </div>
        <EventList events={state.events} />
      </section>
    </div>
  );
}
