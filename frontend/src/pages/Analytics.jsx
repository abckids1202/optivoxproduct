import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useEffect, useState } from "react";
import { Activity, CalendarCheck, Clock3, Users } from "lucide-react";
import StatCard from "../components/StatCard";
import { fetchAnalytics } from "../services/api";

const colors = ["#2dd4bf", "#f59e0b", "#ef4444", "#60a5fa"];

export default function Analytics({ state }) {
  const [data, setData] = useState(state.analytics || {});
  useEffect(() => { fetchAnalytics().then(setData).catch(() => setData(state.analytics || {})); }, [state.analytics]);
  const attendanceDays = data.attendanceByDay || [];
  const byStatus = data.byStatus || [];
  const methodSplit = data.methodSplit || [];
  const roster = data.rosterTotals || { registered: state.summary.registeredPeople || 0, active: 0, seenToday: state.summary.presentToday || 0 };
  const present = byStatus.find((item) => item.name === "Present")?.value || state.summary.presentToday || 0;
  const late = byStatus.find((item) => item.name === "Late")?.value || 0;
  const lateRate = present + late ? Math.round((late / (present + late)) * 100) : 0;
  const absenceTotal = (data.absenceSplit || []).reduce((sum, item) => sum + Number(item.value || 0), 0);
  const openIncidents = data.incidentTotals?.open || state.summary?.openIncidents || 0;
  const classCoverage = data.classCoverage || [];
  const severityRows = (data.bySeverity || []).map((item) => ({
    label: severityLabel(item.severity),
    count: Number(item.count || 0),
  }));
  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Roster coverage" value={`${roster.seenToday}/${roster.registered}`} detail="Registered people seen today" icon={Users} tone="info" />
        <StatCard label="Attendance rate" value={`${roster.registered ? Math.round((present / roster.registered) * 100) : 0}%`} detail="Present against roster" icon={CalendarCheck} tone="success" />
        <StatCard label="Late rate" value={`${lateRate}%`} detail="Among clocked-in records" icon={Clock3} tone="warning" />
        <StatCard label="Tracked events" value={data.totalEvents || state.events.length} detail="Stored security observations" icon={Activity} tone="neutral" />
        <StatCard label="Open incidents" value={openIncidents} detail="Human review queue" icon={Activity} tone="danger" />
        <StatCard label="Absence records" value={absenceTotal} detail="Official and inferred" icon={CalendarCheck} tone="warning" />
      </div>
      <div className="analytics-grid">
      <ChartPanel title="Attendance trend · people per day" note="Present and late records from the attendance table.">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={attendanceDays}>
            <CartesianGrid strokeDasharray="3 3" stroke="#263244" />
            <XAxis dataKey="day" stroke="#91a3b8" />
            <YAxis stroke="#91a3b8" />
            <Tooltip />
            <Bar dataKey="present" stackId="a" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
            <Bar dataKey="late" stackId="a" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Events by hour · stored observations" note="Use this to identify busy review windows, not attendance volume.">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.eventsByHour || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#263244" />
            <XAxis dataKey="hour" stroke="#91a3b8" />
            <YAxis stroke="#91a3b8" />
            <Tooltip />
            <Bar dataKey="events" fill="#60a5fa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Attendance status mix">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart><Pie data={byStatus} dataKey="value" nameKey="name" outerRadius={94}>{byStatus.map((entry, index) => <Cell key={entry.name} fill={colors[index % colors.length]} />)}</Pie><Tooltip /></PieChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Security categories · event count">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={data.securityCategories || []} dataKey="value" nameKey="name" outerRadius={94}>
              {(data.securityCategories || []).map((entry, index) => (
                <Cell key={entry.name} fill={colors[index % colors.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Attendance method mix">
        <ResponsiveContainer width="100%" height={260}><BarChart data={methodSplit}><CartesianGrid strokeDasharray="3 3" stroke="#263244" /><XAxis dataKey="name" stroke="#91a3b8" /><YAxis stroke="#91a3b8" /><Tooltip /><Bar dataKey="value" fill="#2dd4bf" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Recognition decision mix" note="Evidence decisions, not model accuracy. Unresolved and spoof-suspect states remain in review.">
        <ResponsiveContainer width="100%" height={260}><BarChart data={data.recognitionDecisions || []}><CartesianGrid strokeDasharray="3 3" stroke="#263244" /><XAxis dataKey="name" stroke="#91a3b8" /><YAxis stroke="#91a3b8" /><Tooltip /><Bar dataKey="value" fill="#60a5fa" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Absence status mix" note="Inferred absence is a planning signal until an operator confirms it.">
        <ResponsiveContainer width="100%" height={260}><PieChart><Pie data={data.absenceSplit || []} dataKey="value" nameKey="name" outerRadius={94}>{(data.absenceSplit || []).map((entry, index) => <Cell key={entry.name} fill={colors[index % colors.length]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer>
      </ChartPanel>

      <section className="panel analytics-table-panel">
        <div className="section-heading"><div><p className="eyebrow">Roster comparison</p><h2>Class attendance coverage</h2><p className="chart-note">People with an attendance record in the selected 30-day window.</p></div></div>
        <div className="table-wrap compact-table-wrap"><table><thead><tr><th>Class</th><th>Present</th><th>Roster</th><th>Coverage</th></tr></thead><tbody>{classCoverage.map((row) => <tr key={row.name}><td><strong>{row.name}</strong></td><td>{row.present}</td><td>{row.roster}</td><td>{row.roster ? `${Math.round((row.present / row.roster) * 100)}%` : "0%"}</td></tr>)}</tbody></table>{!classCoverage.length && <p className="empty-copy table-empty">No class metadata is available yet.</p>}</div>
      </section>

      <section className="panel analytics-table-panel">
        <div className="section-heading"><div><p className="eyebrow">Security workload</p><h2>Severity mix</h2><p className="chart-note">Stored security observations by severity; this is not a count of unique incidents.</p></div></div>
        <div className="table-wrap compact-table-wrap"><table><thead><tr><th>Severity</th><th>Observations</th><th>Share</th></tr></thead><tbody>{severityRows.map((row) => <tr key={row.label}><td><span className={`table-status ${row.label.toLowerCase()}`}>{row.label}</span></td><td>{row.count}</td><td>{data.totalEvents ? `${Math.round((row.count / data.totalEvents) * 100)}%` : "0%"}</td></tr>)}</tbody></table>{!severityRows.length && <p className="empty-copy table-empty">No security severity data is available yet.</p>}</div>
      </section>

      <section className="panel analytics-summary">
        <p className="eyebrow">What this proves</p>
        <h2>Demo readiness</h2>
        <p>
          The dashboard separates official attendance from passive presence, then connects both to security history.
        </p>
        <ul>
          <li>Attendance logs use registered identities only; unknown presence is excluded.</li>
          <li>The roster matrix distinguishes no record from late and completed attendance.</li>
          <li>Confidence is recognition evidence, not a measured accuracy guarantee.</li>
        </ul>
      </section>
    </div>
    </div>
  );
}

function severityLabel(value) {
  const numeric = Number(value);
  if (numeric >= 3) return "Critical";
  if (numeric === 2) return "Warning";
  if (numeric === 1) return "Attention";
  return "Normal";
}

function ChartPanel({ title, note, children }) {
  return (
    <section className="panel chart-panel">
      <h2>{title}</h2>
      {note && <p className="chart-note">{note}</p>}
      {children}
    </section>
  );
}
