import { Bell, Gauge, ShieldCheck, UserCheck, UserRoundX, Users } from "lucide-react";
import EventList from "../components/EventList";
import LiveFrame from "../components/LiveFrame";
import PersonCard from "../components/PersonCard";
import StatCard from "../components/StatCard";

export default function Overview({ state, connection }) {
  const summary = state.summary;

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
