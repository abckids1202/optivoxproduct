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

const colors = ["#2dd4bf", "#f59e0b", "#ef4444", "#60a5fa"];

export default function Analytics({ state }) {
  return (
    <div className="analytics-grid">
      <ChartPanel title="Attendance, last seven days">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={state.analytics.attendanceByDay}>
            <CartesianGrid strokeDasharray="3 3" stroke="#263244" />
            <XAxis dataKey="day" stroke="#91a3b8" />
            <YAxis stroke="#91a3b8" />
            <Tooltip />
            <Bar dataKey="present" stackId="a" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
            <Bar dataKey="late" stackId="a" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Events by hour today">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={state.analytics.eventsByHour}>
            <CartesianGrid strokeDasharray="3 3" stroke="#263244" />
            <XAxis dataKey="hour" stroke="#91a3b8" />
            <YAxis stroke="#91a3b8" />
            <Tooltip />
            <Bar dataKey="events" fill="#60a5fa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel title="Security categories">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={state.analytics.securityCategories} dataKey="value" nameKey="name" outerRadius={94}>
              {state.analytics.securityCategories.map((entry, index) => (
                <Cell key={entry.name} fill={colors[index % colors.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </ChartPanel>

      <section className="panel analytics-summary">
        <p className="eyebrow">What this proves</p>
        <h2>Demo readiness</h2>
        <p>
          The dashboard separates official attendance from passive presence, then connects both to security history.
        </p>
        <ul>
          <li>Attendance logs use registered identities only.</li>
          <li>Unknown people remain security or presence records.</li>
          <li>Events can be reviewed with snapshots once the backend serves them.</li>
        </ul>
      </section>
    </div>
  );
}

function ChartPanel({ title, children }) {
  return (
    <section className="panel chart-panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}
