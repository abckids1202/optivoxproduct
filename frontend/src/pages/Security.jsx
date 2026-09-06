import { AlertTriangle, ArrowUpRight, Camera, Check, Filter, ShieldAlert, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import EventList from "../components/EventList";
import StatCard from "../components/StatCard";
import { assignIncident, fetchIncident, reviewEvent, reviewIncident } from "../services/api";

export default function Security({ state }) {
  const [filter, setFilter] = useState("all");
  const [reviewedIds, setReviewedIds] = useState([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState(null);
  const [incidentDetail, setIncidentDetail] = useState(null);
  const [assignee, setAssignee] = useState("");
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

  useEffect(() => {
    let active = true;
    if (selectedIncidentId == null) {
      setIncidentDetail(null);
      return undefined;
    }
    fetchIncident(selectedIncidentId).then((next) => {
      if (active) {
        setIncidentDetail(next);
        setAssignee(next?.assignedTo || "");
      }
    }).catch(() => active && setIncidentDetail(null));
    return () => { active = false; };
  }, [selectedIncidentId]);

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
      const next = await reviewIncident(id, action);
      if (selectedIncidentId === id) setIncidentDetail(next);
    } catch (error) {
      window.alert(error.message);
    }
  }

  async function saveAssignment() {
    if (selectedIncidentId == null) return;
    try {
      setIncidentDetail(await assignIncident(selectedIncidentId, assignee));
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
                <div className="event-meta"><span>Last seen {incident.last_event_at}</span><span>Severity {incident.severity}</span><span>{incident.cameraId || "No camera"}{incident.location ? ` · ${incident.location}` : ""}</span></div>
              </div>
              <div className="incident-actions"><button type="button" onClick={() => setSelectedIncidentId(incident.id)}>View timeline</button>
              {!['dismissed', 'resolved'].includes(incident.status) && <>
                <button type="button" onClick={() => reviewOneIncident(incident.id, "confirm")}><Check size={14} /> Confirm</button>
                <button type="button" onClick={() => reviewOneIncident(incident.id, "resolve")}><ArrowUpRight size={14} /> Resolve</button>
                <button type="button" onClick={() => reviewOneIncident(incident.id, "dismiss")}><X size={14} /> Dismiss</button>
              </>}</div>
            </article>
          ))}
          {!incidents.length && <p className="empty-copy">No grouped incidents have been generated.</p>}
        </div>
      </section>

      {selectedIncidentId != null && <section className="panel incident-detail-panel">
        {!incidentDetail && <p className="empty-copy">Loading incident evidence and review history.</p>}
        {incidentDetail && <>
          <div className="section-heading"><div><p className="eyebrow">Evidence timeline</p><h2>{incidentDetail.summary}</h2><p className="panel-note">Unknown presence, identity uncertainty, and confirmed danger remain separate signals.</p></div><button type="button" className="button-quiet" onClick={() => setSelectedIncidentId(null)}>Close timeline</button></div>
          <div className="incident-context"><span>{incidentDetail.category}</span><span>{incidentDetail.cameraId || "Camera unknown"}</span><span>{incidentDetail.location || "Location unknown"}</span><span>{incidentDetail.event_count} observations</span></div>
          <div className="incident-operator-row"><label><span>Assigned operator</span><input value={assignee} onChange={(event) => setAssignee(event.target.value)} placeholder="e.g. security lead" /></label><button type="button" onClick={saveAssignment}>Save assignment</button></div>
          <div className="incident-timeline">{(incidentDetail.events || []).map((event) => <div className="timeline-item" key={event.id}><div><strong>{event.event_type}</strong><span>{event.timestamp} · {event.camera || "camera unknown"}</span></div>{event.snapshot_url && <a href={event.snapshot_url} target="_blank" rel="noreferrer">Open evidence</a>}</div>)}</div>
          <div className="review-history">{(incidentDetail.review_actions || []).map((action) => <div key={action.id}><strong>{action.action}</strong><span>{action.actor_id || "operator"} · {action.created_at}{action.note ? ` · ${action.note}` : ""}</span></div>)}</div>
        </>}
      </section>}
    </div>
  );
}
