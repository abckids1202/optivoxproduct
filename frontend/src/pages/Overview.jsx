import { Activity, AlertCircle, Bell, ClipboardCheck, Gauge, ShieldCheck, UserCheck, UserPlus, UserRoundX, Users } from "lucide-react";
import EventList from "../components/EventList";
import LiveFrame from "../components/LiveFrame";
import PersonCard from "../components/PersonCard";
import StatCard from "../components/StatCard";

export default function Overview({ state, connection, onNavigate }) {
  const summary = state.summary;
  const attention = state.events.filter((event) => event.reviewed === false || event.severity === "Critical");
  const vehicles = state.vehicles?.active_tracks || [];
  const crowd = state.crowd || {};
  const demographics = state.demographics || {};

  return (
    <div className="page-stack">
      <div className="stats-grid">
        <StatCard label="Present today" value={summary.presentToday} detail="Registered attendance" icon={UserCheck} tone="success" />
        <StatCard label="Visible now" value={summary.visibleNow} detail="Presence detection" icon={Users} tone="info" />
        <StatCard label="Unknown today" value={summary.unknownToday} detail="Not official attendance" icon={UserRoundX} tone="warning" />
        <StatCard label="Security events" value={summary.securityEvents} detail="Needs review" icon={ShieldCheck} tone="danger" />
        <StatCard label="Alerts sent" value={summary.alertsSent} detail="Configured channels" icon={Bell} tone="neutral" />
        <StatCard label="Processing FPS" value={state.engine.fps} detail={state.engine.uptime} icon={Gauge} tone="neutral" />
      </div>

      <div className="command-row">
        <section className="panel attention-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Operator queue</p>
              <h2>Requires attention <span className="count-badge">{attention.length}</span></h2>
            </div>
            <AlertCircle size={20} className="attention-icon" />
          </div>
          {attention.length ? attention.slice(0, 3).map((event) => (
            <button className="attention-item" key={event.id} type="button" onClick={() => onNavigate("security")}>
              <span className={`attention-dot ${event.severity === "Critical" ? "critical" : "warning"}`} />
              <span><strong>{event.type.replaceAll("_", " ")}</strong><small>{event.person} · {event.time}</small></span>
              <span className="attention-arrow">View</span>
            </button>
          )) : <p className="empty-copy">No unresolved events in the current feed.</p>}
        </section>
        <section className="panel quick-panel">
          <div className="section-heading"><div><p className="eyebrow">Shortcuts</p><h2>Demo controls</h2></div></div>
          <div className="quick-actions">
            <button type="button" onClick={() => onNavigate("people")}><UserPlus size={16} /> Register person</button>
            <button type="button" onClick={() => onNavigate("attendance")}><ClipboardCheck size={16} /> Open attendance</button>
            <button type="button" onClick={() => onNavigate("security")}><ShieldCheck size={16} /> Review security</button>
          </div>
        </section>
      </div>

      <div className="overview-grid">
        <LiveFrame engine={state.engine} connection={connection} />

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Presence</p>
              <h2>Currently visible</h2>
            </div>
          </div>
          <div className="person-list">
            {state.visiblePeople.map((person) => (
              <PersonCard key={person.id} person={person} />
            ))}
          </div>
        </section>
      </div>

      <div className="lower-grid">
        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Objects</p>
              <h2>Detected objects</h2>
            </div>
          </div>
          <div className="object-chips">
            {state.objects.map((object) => (
              <div className="object-chip" key={object.name}>
                <strong>{object.name}</strong>
                <span>x{object.count}</span>
                <small>{Math.round(object.confidence * 100)}%</small>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Timeline</p>
              <h2>Latest events</h2>
            </div>
          </div>
          <EventList events={state.events.slice(0, 4)} compact />
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Vehicle intelligence</p>
              <h2>Tracked vehicles</h2>
            </div>
          </div>
          {vehicles.length ? vehicles.slice(0, 6).map((vehicle) => (
            <div className="system-row" key={vehicle.track_id}>
              <span>{vehicle.class_name} · #{vehicle.track_id}</span>
              <strong>{vehicle.speed_kmh == null ? "Speed unavailable" : `${vehicle.speed_kmh} km/h`}</strong>
            </div>
          )) : <p className="empty-copy">No vehicle tracks in the latest fresh detection.</p>}
          <p className="panel-note">Physical speed requires camera calibration. Plate reading is not configured.</p>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Crowd intelligence</p>
              <h2>Occupancy and movement</h2>
            </div>
            <Activity size={19} />
          </div>
          <div className="system-grid">
            <div className="system-row"><span>Tracked people</span><strong>{crowd.people_count ?? state.visiblePeople.length}</strong></div>
            <div className="system-row"><span>Movement</span><strong>{crowd.average_speed_px_s == null ? "Not measured" : `${crowd.average_speed_px_s} px/s`}</strong></div>
            <div className="system-row"><span>Growth rate</span><strong>{crowd.growth_per_minute == null ? "Not measured" : `${crowd.growth_per_minute} / min`}</strong></div>
            <div className="system-row"><span>Capability</span><strong>{crowd.status || "Not reported"}</strong></div>
          </div>
          <p className="panel-note">Current density uses tracked occupancy. Perspective-correct physical density is not configured.</p>
        </section>

        <section className="panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Demographic signal</p>
              <h2>Age bands, aggregate only</h2>
            </div>
          </div>
          <div className="system-grid">
            <div className="system-row"><span>Capability</span><strong>{demographics.status || "Not reported"}</strong></div>
            <div className="system-row"><span>Samples</span><strong>{demographics.samples ?? 0}</strong></div>
            {Object.entries(demographics.age_bands || {}).map(([band, count]) => (
              <div className="system-row" key={band}><span>{band}</span><strong>{count}</strong></div>
            ))}
          </div>
          <p className="panel-note">Approximate age bands are for aggregate analytics only. They do not affect attendance, access, or security decisions.</p>
        </section>
      </div>
    </div>
  );
}
