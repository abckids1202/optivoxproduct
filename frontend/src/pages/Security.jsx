import { AlertTriangle, ArrowUpRight, Camera, Check, Filter, ShieldAlert, X } from "lucide-react";
import { useMemo, useState } from "react";
import EventList from "../components/EventList";
import StatCard from "../components/StatCard";
import { reviewEvent, reviewIncident } from "../services/api";

export default function Security({ state }) {
  const [filter, setFilter] = useState("all");
  const [reviewedIds, setReviewedIds] = useState([]);
  const critical = state.events.filter((e) => e.severity === "Critical").length;
  const warning = state.events.filter((e) => e.severity === "Warning" || e.severity === "Attention").length;
  const unknown = state.events.filter((e) => e.type.includes("UNKNOWN")).length;
  const snapshots = state.events.filter((e) => e.severity !== "Normal").length;
  const incidents = state.incidents || [];
  const openIncidents = incidents.filter((incident) => !["dismissed", "resolved"].includes(incident.status));
  const events = useMemo(() => state.events.filter((event) => {
    const reviewed = reviewedIds.includes(event.id) || event.reviewed;
    return filter === "all" || (filter === "open" && !reviewed) || (filter === "reviewed" && reviewed);
  }), [state.events, filter, reviewedIds]);

  async function reviewOne(id, action) {
    try {
      await reviewEvent(id, action);
      setReviewedIds((current) => [...new Set([...current, id])]);
    } catch (error) {
      window.alert(error.message);
    }
  }

  async function reviewOneIncident(id, action) {
    try {
      await reviewIncident(id, action);
    } catch (error) {
      window.alert(error.message);
    }
  }

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
        <EventList events={events} onReview={reviewOne} />
        {events.length > 0 && <div className="review-strip"><span>Review decisions are persisted with the event audit trail.</span><span>{openIncidents.length} open incident{openIncidents.length === 1 ? "" : "s"}</span></div>}
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Grouped response queue</p>
            <h2>Incidents</h2>
          </div>
          <span className="status-badge tone-info">{incidents.length} grouped</span>
        </div>
        <div className="incident-list">
          {incidents.map((incident) => (
            <article className="incident-item" key={incident.id}>
              <div>
                <div className="event-title"><strong>{incident.summary}</strong><span className={`status-badge ${incident.status === "open" ? "tone-warning" : "tone-success"}`}>{incident.status}</span></div>
                <p>{incident.category} · {incident.event_count} linked observations</p>
                <div className="event-meta"><span>Last seen {incident.last_event_at}</span><span>Severity {incident.severity}</span></div>
              </div>
              {!['dismissed', 'resolved'].includes(incident.status) && <div className="incident-actions">
                <button type="button" onClick={() => reviewOneIncident(incident.id, "confirm")}><Check size={14} /> Confirm</button>
                <button type="button" onClick={() => reviewOneIncident(incident.id, "resolve")}><ArrowUpRight size={14} /> Resolve</button>
                <button type="button" onClick={() => reviewOneIncident(incident.id, "dismiss")}><X size={14} /> Dismiss</button>
              </div>}
            </article>
          ))}
          {!incidents.length && <p className="empty-copy">No grouped incidents have been generated.</p>}
        </div>
      </section>
    </div>
  );
}
