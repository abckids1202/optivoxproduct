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
  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Roster coverage" value={`${roster.seenToday}/${roster.registered}`} detail="Registered people seen today" icon={Users} tone="info" />
        <StatCard label="Attendance rate" value={`${roster.registered ? Math.round((present / roster.registered) * 100) : 0}%`} detail="Present against roster" icon={CalendarCheck} tone="success" />
        <StatCard label="Late rate" value={`${lateRate}%`} detail="Among clocked-in records" icon={Clock3} tone="warning" />
        <StatCard label="Tracked events" value={data.totalEvents || state.events.length} detail="Stored security observations" icon={Activity} tone="neutral" />
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

function ChartPanel({ title, note, children }) {
  return (
    <section className="panel chart-panel">
      <h2>{title}</h2>
      {note && <p className="chart-note">{note}</p>}
      {children}
    </section>
  );
}
