import { AlertCircle, Bell, ClipboardCheck, Gauge, ShieldCheck, UserCheck, UserPlus, UserRoundX, Users } from "lucide-react";
import EventList from "../components/EventList";
import LiveFrame from "../components/LiveFrame";
import PersonCard from "../components/PersonCard";
import StatCard from "../components/StatCard";

export default function Overview({ state, connection, onNavigate }) {
  const summary = state.summary;
  const attention = state.events.filter((event) => event.reviewed === false || event.severity === "Critical");

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
      </div>
    </div>
  );
}
