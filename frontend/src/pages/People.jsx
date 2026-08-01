import { Search, Play, UserPlus, Users } from "lucide-react";
import { useMemo, useState } from "react";
import { sendCommand } from "../services/api";

export default function People({ state }) {
  const [query, setQuery] = useState("");
  const people = useMemo(() => (state.people || []).filter((person) => `${person.name} ${person.role || ""} ${person.className || ""} ${(person.subjects || []).join(" ")}`.toLowerCase().includes(query.toLowerCase())), [state.people, query]);
  async function startEnrollment(mode) {
    const name = window.prompt("Enter the person's name for local enrollment");
    if (!name) return;
    const role = window.prompt("Role or class (optional)") || "";
    const consent = window.confirm("Confirm that consent was collected for this enrollment.");
    if (!consent) return;
    const command = mode === "visible" ? "register_visible_unknown" : "start_enrollment";
    try {
      await sendCommand(command, { name, role, consent_confirmed: true });
      window.alert("Enrollment command queued. Keep the person visible to the local OptiVox engine.");
    } catch (error) {
      window.alert(error.message);
    }
  }

  return (
    <div className="people-layout">
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Enrollment</p>
            <h2>Registration modes</h2>
          </div>
        </div>

        <div className="enrollment-actions">
          <button type="button" onClick={() => startEnrollment("new")}>
            <UserPlus size={18} />
            Register new person
          </button>
          <button type="button" onClick={() => startEnrollment("visible")}>
            <Play size={18} />
            Register visible unknown
          </button>
        </div>

        <div className="flow-list">
          <div><strong>Registered Attendance</strong><span>Admin captures face locally, then Optivox can clock attendance automatically.</span></div>
          <div><strong>Passive Presence</strong><span>Unknown people appear in live security state without becoming official attendance.</span></div>
          <div><strong>Web Command</strong><span>The website asks the local engine to enroll. The browser never opens the webcam separately.</span></div>
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Face database</p>
            <h2>Registered people</h2>
          </div>
          <label className="search-box"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search full roster" aria-label="Search people" /></label>
        </div>
        <div className="roster-strip"><span><Users size={15} /> <strong>{state.people?.length || 0}</strong> registered</span><span><strong>{state.people?.filter((person) => person.active !== false).length || 0}</strong> active profiles</span><span><strong>{state.people?.filter((person) => person.status !== "Not Yet Detected").length || 0}</strong> seen today</span></div>
        <div className="people-grid">
          {people.map((person) => (
            <article className="profile-card" key={person.id}>
              <div className="avatar">{person.name.slice(0, 1)}</div>
              <div>
                <strong>{person.name}</strong>
                <span>{person.role || "Not assigned"}{person.className ? ` · ${person.className}` : ""}</span>
                <small>{person.samples} samples · last seen {person.lastSeen || "Never"}</small>
                <div className="profile-subjects">{(person.subjects || []).length ? person.subjects.map((subject) => <span key={subject}>{subject}</span>) : <em>No subjects assigned</em>}</div>
              </div>
              <em>{person.status}</em>
            </article>
          ))}
        </div>
        {!people.length && <p className="empty-copy">No registered people match this search.</p>}
      </section>
    </div>
  );
}
