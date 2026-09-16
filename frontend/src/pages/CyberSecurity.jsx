import { Activity, AlertOctagon, Check, Database, FileWarning, KeyRound, LockKeyhole, Radio, ShieldCheck, UserCog, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import StatCard from "../components/StatCard";
import {
  assignCyberIncident,
  fetchCyberAdminHistory,
  fetchCyberAlertFailures,
  fetchCyberAuthHistory,
  fetchCyberIncident,
  fetchCyberIncidents,
  fetchCyberIntegrityFailures,
  fetchCyberSessions,
  fetchCyberSummary,
  reviewCyberIncident,
} from "../services/api";

const tabs = [
  ["incidents", "Incident queue", AlertOctagon],
  ["sessions", "Active sessions", Radio],
  ["auth", "Authentication", KeyRound],
  ["admin", "Admin actions", UserCog],
  ["integrity", "Integrity and delivery", FileWarning],
  ["device", "Device health", Radio],
];

export default function CyberSecurity({ state }) {
  const [tab, setTab] = useState("incidents");
  const [summary, setSummary] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [authHistory, setAuthHistory] = useState([]);
  const [adminHistory, setAdminHistory] = useState([]);
  const [integrity, setIntegrity] = useState([]);
  const [alertFailures, setAlertFailures] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filters, setFilters] = useState({ status: "all", category: "all", severity: "all" });
  const [error, setError] = useState("");

  async function load() {
    try {
      const [nextSummary, nextIncidents, nextSessions, nextAuth, nextAdmin, nextIntegrity, nextFailures] = await Promise.all([
        fetchCyberSummary(),
        fetchCyberIncidents(filters),
        fetchCyberSessions(),
        fetchCyberAuthHistory(),
        fetchCyberAdminHistory(),
        fetchCyberIntegrityFailures(),
        fetchCyberAlertFailures(),
      ]);
      setSummary(nextSummary);
      setIncidents(nextIncidents || []);
      setSessions(nextSessions || []);
      setAuthHistory(nextAuth || []);
      setAdminHistory(nextAdmin || []);
      setIntegrity(nextIntegrity || []);
      setAlertFailures(nextFailures || []);
      setError("");
    } catch (nextError) {
      setError(nextError.message || "Cybersecurity data is unavailable.");
    }
  }

  useEffect(() => { load(); }, [filters.status, filters.category, filters.severity]);

  async function review(id, action) {
    try {
      const next = await reviewCyberIncident(id, action);
      setSelected(next);
      await load();
    } catch (nextError) { window.alert(nextError.message); }
  }

  async function assign(id) {
    const assignee = window.prompt("Assign this incident to:", selected?.assignedTo || "");
    if (assignee === null) return;
    try {
      setSelected(await assignCyberIncident(id, assignee));
      await load();
    } catch (nextError) { window.alert(nextError.message); }
  }

  const categories = useMemo(() => [...new Set((summary?.categories || []).map((item) => item.name))], [summary]);
  const openCount = Number(summary?.openIncidents || 0);
  const severity = (value) => Number(value) >= 3 ? "tone-danger" : Number(value) >= 2 ? "tone-warning" : "tone-info";

  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Open cyber incidents" value={openCount} detail="Needs operational review" icon={AlertOctagon} tone={openCount ? "danger" : "success"} />
        <StatCard label="Cyber events" value={summary?.eventTotal ?? "--"} detail="Application and edge telemetry" icon={Activity} tone="info" />
        <StatCard label="Active sessions" value={sessions.length} detail="Authenticated operators" icon={Radio} tone="neutral" />
        <StatCard label="Integrity or delivery" value={integrity.length + alertFailures.length} detail="Recorded failures" icon={ShieldCheck} tone={integrity.length + alertFailures.length ? "warning" : "success"} />
      </div>

      <section className="panel cyber-ops-header">
        <div>
          <p className="eyebrow">Application security operations</p>
          <h2>Cyber defense queue</h2>
          <p className="panel-note">Separate from physical camera incidents. Each record is correlated to a source event, audit reference, alert attempt, and review state.</p>
        </div>
        <div className="cyber-ops-badge"><LockKeyhole size={17} /> {summary?.physicalEventsSeparate ? "Physical events separated" : "Separation unavailable"}</div>
      </section>

      {error && <div className="banner">{error}</div>}

      <div className="cyber-tabs" role="tablist" aria-label="Cybersecurity operations views">
        {tabs.map(([id, label, Icon]) => <button type="button" role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} key={id} onClick={() => setTab(id)}><Icon size={16} />{label}</button>)}
      </div>

      {tab === "incidents" && <IncidentQueue incidents={incidents} filters={filters} setFilters={setFilters} categories={categories} severity={severity} onSelect={async (id) => { try { setSelected(await fetchCyberIncident(id)); } catch (nextError) { window.alert(nextError.message); } }} onReview={review} onAssign={assign} />}
      {tab === "sessions" && <DataTable title="Authenticated sessions" icon={Radio} columns={["User", "Role", "IP address", "User agent", "Last seen", "Expires"]} rows={sessions.map((row) => [row.username, row.role, row.client_ip || "local", row.user_agent || "Not reported", row.last_seen_at, row.expires_at])} empty="No active authenticated sessions." />}
      {tab === "auth" && <AuditTable title="Authentication history" icon={KeyRound} rows={authHistory} empty="No authentication actions recorded." />}
      {tab === "admin" && <AuditTable title="Administrative action history" icon={UserCog} rows={adminHistory} empty="No administrative actions recorded." />}
      {tab === "integrity" && <IntegrityView integrity={integrity} failures={alertFailures} />}
      {tab === "device" && <DeviceHealth state={state} />}

      {selected && <IncidentDetail incident={selected} severity={severity} onClose={() => setSelected(null)} onReview={review} onAssign={assign} />}
    </div>
  );
}

function IncidentQueue({ incidents, filters, setFilters, categories, severity, onSelect, onReview, onAssign }) {
  return <section className="panel">
    <div className="section-heading"><div><p className="eyebrow">Correlation and response</p><h2>Cybersecurity incidents</h2></div><span className="status-badge tone-info">{incidents.length} shown</span></div>
    <div className="incident-filters cyber-filters">
      <label><span>State</span><select value={filters.status} onChange={(event) => setFilters({ ...filters, status: event.target.value })}><option value="all">All states</option>{["open", "acknowledged", "assigned", "escalated", "confirmed", "dismissed", "resolved"].map((value) => <option key={value}>{value}</option>)}</select></label>
      <label><span>Category</span><select value={filters.category} onChange={(event) => setFilters({ ...filters, category: event.target.value })}><option value="all">All categories</option>{categories.map((value) => <option key={value}>{value}</option>)}</select></label>
      <label><span>Severity</span><select value={filters.severity} onChange={(event) => setFilters({ ...filters, severity: event.target.value })}><option value="all">All levels</option><option value="1">1 · Notice</option><option value="2">2 · Warning</option><option value="3">3 · Critical</option></select></label>
    </div>
    <div className="incident-list">
      {incidents.map((incident) => <article className="incident-item" key={incident.id}><div><div className="event-title"><strong>{incident.summary}</strong><span className={`status-badge ${severity(incident.severity)}`}>S{incident.severity} · {incident.status}</span></div><p>{incident.category} · {incident.eventCount} event{incident.eventCount === 1 ? "" : "s"} · {incident.alertCount} alert attempt{incident.alertCount === 1 ? "" : "s"}</p><div className="event-meta"><span>{incident.lastEventAt}</span><span>{incident.source}</span><span>{incident.actorId || "system"}</span><span>{incident.ipAddress || "local"}</span></div></div><div className="incident-actions"><button type="button" onClick={() => onSelect(incident.id)}>Open detail</button>{!['dismissed', 'resolved'].includes(incident.status) && <><button type="button" onClick={() => onAssign(incident.id)}>Assign</button>{incident.status === "open" && <button type="button" onClick={() => onReview(incident.id, "acknowledge")}>Acknowledge</button>}<button type="button" onClick={() => onReview(incident.id, "escalate")}><AlertOctagon size={14} /> Escalate</button><button type="button" onClick={() => onReview(incident.id, "resolve")}><Check size={14} /> Resolve</button><button type="button" onClick={() => onReview(incident.id, "dismiss")}><X size={14} /> Dismiss</button></>}</div></article>)}
      {!incidents.length && <p className="empty-copy">No cybersecurity incidents match the current filters.</p>}
    </div>
  </section>;
}

function IncidentDetail({ incident, severity, onClose, onReview, onAssign }) {
  return <section className="panel incident-detail-panel"><div className="section-heading"><div><p className="eyebrow">Incident trace</p><h2>{incident.summary}</h2></div><button type="button" className="button-quiet" onClick={onClose}>Close detail</button></div><div className="incident-context"><span className={`status-badge ${severity(incident.severity)}`}>Severity {incident.severity}</span><span>{incident.status}</span><span>{incident.category}</span><span>{incident.source}</span><span>Correlation {String(incident.correlationKey || "").slice(0, 42)}</span></div><div className="incident-records"><div><p className="eyebrow">Identity and origin</p><div className="detail-row"><strong>Actor</strong><span>{incident.actorId || "system"}</span></div><div className="detail-row"><strong>Device</strong><span>{incident.deviceId || "unknown"}</span></div><div className="detail-row"><strong>Network</strong><span>{incident.ipAddress || "local"}</span></div><div className="detail-row"><strong>User agent</strong><span>{incident.userAgent || "Not reported"}</span></div></div><div><p className="eyebrow">Operator response</p><div className="detail-row"><strong>Assigned</strong><span>{incident.assignedTo || "Unassigned"}</span></div><div className="detail-row"><strong>Timeline</strong><span>{incident.firstEventAt} to {incident.lastEventAt}</span></div><div className="incident-actions"><button type="button" onClick={() => onAssign(incident.id)}>Assign</button><button type="button" onClick={() => onReview(incident.id, "confirm")}>Confirm</button><button type="button" onClick={() => onReview(incident.id, "escalate")}>Escalate</button></div></div></div><div className="cyber-event-timeline">{(incident.events || []).map((event) => <div className="timeline-item" key={event.id}><div><strong>{event.eventType}</strong><span>{event.occurredAt} · {event.source} · {event.correlationId}</span><span>{event.ipAddress || "local"} · {event.actorId || "system"}</span></div>{event.evidenceRef && <span className="status-badge tone-info">Log: {event.evidenceRef}</span>}</div>)}</div><div className="incident-records"><DataTable title="Alert attempts" icon={Radio} columns={["Channel", "Status", "Attempted", "Error"]} rows={(incident.alerts || []).map((row) => [row.channel, row.status, row.attempted_at, row.error || "-"])} empty="No alert attempts." /><DataTable title="Review history" icon={UserCog} columns={["Action", "Actor", "Time", "Note"]} rows={(incident.reviews || []).map((row) => [row.action, row.actor_id || "system", row.created_at, row.note || "-"])} empty="No review actions." /></div></section>;
}

function AuditTable({ title, icon: Icon, rows, empty }) {
  return <DataTable title={title} icon={Icon} columns={["Action", "Entity", "Actor", "Time", "Context"]} rows={rows.map((row) => [row.action, `${row.entity_type || ""} ${row.entity_id || ""}`.trim(), row.actor_id || row.actor_type || "system", row.created_at, JSON.stringify(row.details || {})])} empty={empty} />;
}

function IntegrityView({ integrity, failures }) {
  return <div className="cyber-two-column"><DataTable title="Integrity failures" icon={Database} columns={["Type", "Source", "Time", "Evidence"]} rows={integrity.map((row) => [row.eventType, row.source, row.occurredAt, row.evidenceRef || "-"])} empty="No model or database integrity failures." /><DataTable title="Alert delivery failures" icon={FileWarning} columns={["Type", "Source", "Time", "Details"]} rows={failures.map((row) => [row.eventType, row.source, row.occurredAt, row.details?.error || row.evidenceRef || "-"])} empty="No alert delivery failures." /></div>;
}

function DeviceHealth({ state }) {
  const engine = state?.engine || {};
  const events = (state?.healthEvents || []).slice().reverse();
  return <div className="cyber-two-column"><section className="panel"><div className="section-heading"><div><p className="eyebrow">Edge agent status</p><h2>Camera and runtime health</h2></div><Radio size={19} /></div><div className="system-grid"><div className="system-row"><span>Engine</span><strong>{engine.status || "Unknown"}</strong></div><div className="system-row"><span>Camera</span><strong>{engine.camera || "Unknown"}</strong></div><div className="system-row"><span>Location</span><strong>{engine.location || "Unknown"}</strong></div><div className="system-row"><span>Last heartbeat</span><strong>{engine.lastHeartbeat || "Not reported"}</strong></div><div className="system-row"><span>Last error</span><strong>{engine.lastError || "None reported"}</strong></div></div></section><DataTable title="Health transitions" icon={Radio} columns={["Type", "State", "Camera", "Recorded"]} rows={events.map((row) => [row.type || "health", row.state || row.reason || "event", row.camera_id || "local", row.recorded_at || row.timestamp || "-"])} empty="No camera or runtime health transitions recorded." /></div>;
}

function DataTable({ title, icon: Icon, columns, rows, empty }) {
  return <section className="panel"><div className="section-heading"><div><p className="eyebrow">Operations telemetry</p><h2>{title}</h2></div><Icon size={19} aria-hidden="true" /></div>{rows.length ? <div className="table-wrap"><table className="data-table"><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={`${title}-${index}`}>{row.map((value, cell) => <td key={cell}>{String(value ?? "-")}</td>)}</tr>)}</tbody></table></div> : <p className="empty-copy">{empty}</p>}</section>;
}
