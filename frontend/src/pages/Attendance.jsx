import { Clock, Download, Search } from "lucide-react";
import { useMemo, useState } from "react";
import StatCard from "../components/StatCard";

export default function Attendance({ state }) {
  const [query, setQuery] = useState("");
  const rows = state.attendance?.length ? state.attendance : [];
  const filteredRows = useMemo(() => rows.filter((person) => `${person.name} ${person.role} ${person.className}`.toLowerCase().includes(query.toLowerCase())), [rows, query]);
  const present = rows.filter((p) => p.status === "Present" || p.status === "Late").length;
  const late = rows.filter((p) => p.status === "Late").length;
  const left = rows.filter((p) => p.status === "Left").length;
  const pending = state.summary?.not_yet_detected || 0;

  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Currently inside" value={present} detail="Clocked in" icon={Clock} tone="success" />
        <StatCard label="Late today" value={late} detail="After grace time" icon={Clock} tone="warning" />
        <StatCard label="Clocked out" value={left} detail="Left today" icon={Clock} tone="neutral" />
        <StatCard label="Not yet detected" value={pending} detail="Registered roster" icon={Clock} tone="info" />
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Official attendance</p>
            <h2>Daily attendance logs</h2>
          </div>
          <div className="toolbar">
            <label className="search-box">
              <Search size={16} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search person" aria-label="Search attendance" />
            </label>
            <button type="button" onClick={() => exportAttendance(filteredRows)}>
              <Download size={16} />
              Export CSV
            </button>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Person</th>
                <th>Role/Class</th>
                <th>Status</th>
                <th>Clock-in</th>
                <th>Clock-out</th>
                <th>Method</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((person) => (
                <tr key={person.id}>
                  <td>
                    <strong>{person.name}</strong>
                    <span>Last seen {person.lastSeen}</span>
                  </td>
                  <td>{person.role} / {person.className}</td>
                  <td><span className={`table-status ${statusClass(person.status)}`}>{person.status}</span></td>
                  <td>{person.clockIn || "-"}</td>
                  <td>{person.clockOut || "-"}</td>
                  <td>{person.method}</td>
                  <td>{person.confidence ? `${Math.round(person.confidence * 100)}%` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!filteredRows.length && <p className="empty-copy table-empty">No attendance records match this search.</p>}
        </div>
      </section>
    </div>
  );
}

function exportAttendance(rows) {
  const header = ["Name", "Role", "Class", "Status", "Clock in", "Clock out", "Method", "Confidence"];
  const lines = rows.map((person) => [person.name, person.role, person.className, person.status, person.clockIn || "", person.clockOut || "", person.method, person.confidence ? `${Math.round(person.confidence * 100)}%` : ""]);
  const csv = [header, ...lines].map((line) => line.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `optivox-attendance-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function statusClass(status) {
  return String(status).toLowerCase().replaceAll(" ", "-");
}
