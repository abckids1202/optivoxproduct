import { AlertTriangle, Camera, Filter, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
import EventList from "../components/EventList";
import StatCard from "../components/StatCard";

export default function Security({ state }) {
  const [filter, setFilter] = useState("all");
  const [reviewedIds, setReviewedIds] = useState([]);
  const critical = state.events.filter((e) => e.severity === "Critical").length;
  const warning = state.events.filter((e) => e.severity === "Warning" || e.severity === "Attention").length;
  const unknown = state.events.filter((e) => e.type.includes("UNKNOWN")).length;
  const snapshots = state.events.filter((e) => e.severity !== "Normal").length;
  const events = useMemo(() => state.events.filter((event) => {
    const reviewed = reviewedIds.includes(event.id) || event.reviewed;
    return filter === "all" || (filter === "open" && !reviewed) || (filter === "reviewed" && reviewed);
  }), [state.events, filter, reviewedIds]);

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
          <label className="filter-select"><Filter size={16} /><select value={filter} onChange={(event) => setFilter(event.target.value)} aria-label="Filter security events"><option value="all">All events</option><option value="open">Open review</option><option value="reviewed">Reviewed</option></select></label>
          <button type="button" onClick={() => setFilter("open")}>
            <Filter size={16} />
            {state.events.filter((event) => !event.reviewed).length} open
          </button>
        </div>
        <EventList events={events} />
        {events.length > 0 && <div className="review-strip"><span>Review actions are local to this exhibition session.</span><button type="button" onClick={() => setReviewedIds(state.events.map((event) => event.id))}>Mark visible reviewed</button></div>}
      </section>
    </div>
  );
}
