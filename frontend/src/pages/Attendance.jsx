import { CalendarDays, ChevronLeft, ChevronRight, Clock, Download, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { fetchAcademicOverview, fetchAttendanceCalendar } from "../services/api";
import StatCard from "../components/StatCard";

export default function Attendance({ state }) {
  const [query, setQuery] = useState("");
  const [view, setView] = useState("today");
  const [monthDate, setMonthDate] = useState(() => new Date());
  const [calendar, setCalendar] = useState(null);
  const [academic, setAcademic] = useState(null);
  const rows = state.attendance?.length ? state.attendance : [];
  const filteredRows = useMemo(() => rows.filter((person) => `${person.name} ${person.role} ${person.className}`.toLowerCase().includes(query.toLowerCase())), [rows, query]);
  useEffect(() => {
    const year = monthDate.getFullYear();
    const month = monthDate.getMonth() + 1;
    Promise.all([fetchAttendanceCalendar(year, month), fetchAcademicOverview(year, month)]).then(([nextCalendar, nextAcademic]) => { setCalendar(nextCalendar); setAcademic(nextAcademic); }).catch(() => { setCalendar(null); setAcademic(null); });
  }, [monthDate]);
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

      <section className="panel">
        <div className="section-heading">
          <div><p className="eyebrow">Roster history</p><h2>Attendance by day</h2><p className="panel-note">Every registered person is shown, including people not detected this month.</p></div>
          <div className="calendar-controls"><button className="icon-button" type="button" title="Previous month" onClick={() => shiftMonth(setMonthDate, -1)}><ChevronLeft size={17} /></button><strong><CalendarDays size={16} /> {monthDate.toLocaleDateString("en-GB", { month: "long", year: "numeric" })}</strong><button className="icon-button" type="button" title="Next month" onClick={() => shiftMonth(setMonthDate, 1)}><ChevronRight size={17} /></button></div>
        </div>
        <div className="view-tabs"><button className={view === "today" ? "selected" : ""} type="button" onClick={() => setView("today")}>Daily logs</button><button className={view === "month" ? "selected" : ""} type="button" onClick={() => setView("month")}>Monthly matrix</button><button className={view === "subjects" ? "selected" : ""} type="button" onClick={() => setView("subjects")}>Subjects & absences</button></div>
        {view === "today" && <div className="attendance-summary-strip"><span><strong>{present}</strong> present today</span><span><strong>{late}</strong> late</span><span><strong>{left}</strong> clocked out</span><span><strong>{pending}</strong> not detected</span></div>}
        {view === "month" && <AttendanceMatrix calendar={calendar} query={query} />}
        {view === "subjects" && <SubjectAbsencePanel academic={academic} />}
      </section>
    </div>
  );
}

function AttendanceMatrix({ calendar, query }) {
  if (!calendar) return <p className="empty-copy">Loading roster history or waiting for the backend.</p>;
  const visibleDays = calendar.days.filter((day) => new Date(`${day}T00:00:00`).getDay() !== 0 && new Date(`${day}T00:00:00`).getDay() !== 6);
  const people = calendar.people.filter((person) => `${person.name} ${person.role}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="matrix-wrap"><table className="attendance-matrix"><thead><tr><th className="sticky-person">Person</th>{visibleDays.map((day) => <th key={day}>{new Date(`${day}T00:00:00`).getDate()}</th>)}</tr></thead><tbody>{people.map((person) => <tr key={person.id}><td className="sticky-person"><strong>{person.name}</strong><span>{person.role || "Not assigned"}</span></td>{visibleDays.map((day) => <td key={day} title={`${person.name} · ${day}`}>{matrixMark(person.records?.[day])}</td>)}</tr>)}</tbody></table></div>;
}

function matrixMark(record) {
  if (!record) return <span className="matrix-mark absent" aria-label="Absent or no record">×</span>;
  return <span className={`matrix-mark ${record.status === "Late" ? "late" : "present"}`} aria-label={record.status}>{record.status === "Late" ? "L" : "✓"}</span>;
}

function SubjectAbsencePanel({ academic }) {
  if (!academic) return <p className="empty-copy">Loading academic profile data.</p>;
  return <div className="academic-grid"><div><p className="eyebrow">Subjects in roster metadata</p><div className="subject-chips">{academic.subjects.length ? academic.subjects.map((subject) => <span className="subject-chip" key={subject}>{subject}</span>) : <span className="empty-copy">No subjects assigned yet.</span>}</div></div><div><p className="eyebrow">Person subject assignments</p><div className="subject-list">{academic.profiles.map((profile) => <div key={profile.person_id}><strong>{profile.name}</strong><span>{profile.subjects.length ? profile.subjects.join(" · ") : "No subjects assigned"}</span></div>)}</div></div><div className="absence-callout"><p className="eyebrow">Absence list</p><strong>{academic.absence_records.length} recorded exceptions</strong><p>{academic.absence_note}</p></div></div>;
}

function shiftMonth(setter, amount) { setter((current) => new Date(current.getFullYear(), current.getMonth() + amount, 1)); }

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
