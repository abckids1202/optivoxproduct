import { CheckCircle2, Loader2, Play, Search, UserPlus, Users, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { cancelEnrollment, fetchEnrollmentStatus, fetchPerson, sendCommand, updatePerson } from "../services/api";

export default function People({ state }) {
  const [query, setQuery] = useState("");
  const [form, setForm] = useState({ name: "", role: "", consent: false });
  const [enrollment, setEnrollment] = useState({ stage: "idle", message: "No enrollment in progress." });
  const [submitting, setSubmitting] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [profile, setProfile] = useState(null);
  const [profileEdit, setProfileEdit] = useState(null);
  const [profileSaving, setProfileSaving] = useState(false);
  const people = useMemo(() => (state.people || []).filter((person) => `${person.name} ${person.role || ""} ${person.className || ""} ${(person.subjects || []).join(" ")}`.toLowerCase().includes(query.toLowerCase())), [state.people, query]);

  useEffect(() => {
    let active = true;
    let interval;
    async function loadStatus() {
      try {
        const next = await fetchEnrollmentStatus();
        if (active) setEnrollment(next);
      } catch {
        if (active) setEnrollment({ stage: "unknown", message: "Enrollment status is unavailable." });
      }
    }
    loadStatus();
    interval = window.setInterval(loadStatus, 1000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let active = true;
    if (selectedId == null) {
      setProfile(null);
      return undefined;
    }
    fetchPerson(selectedId).then((next) => {
      if (!active) return;
      setProfile(next);
      setProfileEdit({
        name: next?.name || "",
        role: next?.role || "",
        className: next?.className || "",
        studentId: next?.studentId || "",
        subjects: (next?.subjects || []).join(", "),
      });
    }).catch(() => active && setProfile(null));
    return () => { active = false; };
  }, [selectedId]);

  async function startEnrollment(mode) {
    const name = form.name.trim();
    if (!name || !form.consent) return;
    setSubmitting(true);
    try {
      const command = mode === "visible" ? "register_visible_unknown" : "start_enrollment";
      await sendCommand(command, { name, role: form.role.trim(), consent_confirmed: true });
      setEnrollment({ stage: "queued", person_name: name, message: "Command queued. Keep exactly one face visible to the local camera." });
    } catch (error) {
      setEnrollment({ stage: "failed", person_name: name, message: error.message });
    } finally {
      setSubmitting(false);
    }
  }

  async function stopEnrollment() {
    try {
      await cancelEnrollment();
      setEnrollment((current) => ({ ...current, stage: "cancelled", message: "Cancellation queued for the local engine." }));
    } catch (error) {
      setEnrollment((current) => ({ ...current, stage: "failed", message: error.message }));
    }
  }

  async function saveProfile() {
    if (!profileEdit || !selectedId) return;
    setProfileSaving(true);
    try {
      const next = await updatePerson(selectedId, {
        ...profileEdit,
        subjects: profileEdit.subjects.split(",").map((subject) => subject.trim()).filter(Boolean),
      });
      setProfile(next);
    } catch (error) {
      window.alert(error.message);
    } finally {
      setProfileSaving(false);
    }
  }

  const accepted = Number(enrollment.accepted || 0);
  const maximum = Number(enrollment.maximum || 10);
  const progress = enrollment.stage === "completed" ? 100 : Math.min(100, Math.round((accepted / Math.max(1, maximum)) * 100));
  const enrollmentActive = ["queued", "capturing"].includes(enrollment.stage);

  return (
    <div className="people-layout">
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Enrollment</p>
            <h2>Registration modes</h2>
          </div>
        </div>

        <div className="enrollment-form">
          <label>
            <span>Full name</span>
            <input value={form.name} onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))} placeholder="e.g. Jordan Lee" />
          </label>
          <label>
            <span>Role or class <em>optional</em></span>
            <input value={form.role} onChange={(event) => setForm((current) => ({ ...current, role: event.target.value }))} placeholder="Student · 10A" />
          </label>
          <label className="consent-check">
            <input type="checkbox" checked={form.consent} onChange={(event) => setForm((current) => ({ ...current, consent: event.target.checked }))} />
            <span>Consent confirmed for local biometric enrollment</span>
          </label>
          <div className="enrollment-actions">
            <button type="button" disabled={submitting || enrollmentActive || !form.name.trim() || !form.consent} onClick={() => startEnrollment("new")}>
              {submitting ? <Loader2 className="spin" size={18} /> : <UserPlus size={18} />}
              Start guided enrollment
            </button>
            <button type="button" disabled={submitting || enrollmentActive || !form.name.trim() || !form.consent} onClick={() => startEnrollment("visible")}>
              <Play size={18} />
              Register one visible face
            </button>
            {enrollmentActive && <button type="button" className="button-danger" onClick={stopEnrollment}><XCircle size={18} /> Cancel enrollment</button>}
          </div>
        </div>

        <div className={`enrollment-status ${enrollment.stage}`} aria-live="polite">
          <div className="status-line">
            <div>
              <strong>{enrollment.person_name ? `${enrollment.person_name} · ` : ""}{enrollment.stage.replaceAll("_", " ")}</strong>
              <span>{enrollment.message || "Waiting for the local engine."}</span>
            </div>
            {enrollment.stage === "completed" && <CheckCircle2 size={20} />}
          </div>
          {(enrollment.stage === "capturing" || enrollment.stage === "completed") && <>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="progress-meta"><span>{accepted} accepted</span><span>target {enrollment.minimum || 5}-{maximum}</span>{enrollment.quality_score != null && <span>quality {Math.round(enrollment.quality_score)}</span>}</div>
          </>}
        </div>

        <div className="flow-list">
          <div><strong>Registered Attendance</strong><span>Admin captures face locally, then Optivox can clock attendance automatically.</span></div>
          <div><strong>Passive Presence</strong><span>Unknown people appear in live security state without becoming official attendance.</span></div>
          <div><strong>Guided local capture</strong><span>The local engine collects 5-10 quality-gated samples while the dashboard shows progress. The browser never opens the webcam separately.</span></div>
          <div><strong>Attendance safety</strong><span>Only a confirmed, live identity can create official attendance. Unknown, unresolved, and spoof-suspect faces remain presence-only.</span></div>
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
            <article className={`profile-card ${selectedId === person.id ? "selected" : ""}`} key={person.id} onClick={() => setSelectedId(person.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelectedId(person.id); }} role="button" tabIndex={0}>
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

      {selectedId != null && <section className="panel profile-detail-panel">
        {!profile && <p className="empty-copy">Loading the selected student profile.</p>}
        {profile && <>
          <div className="section-heading">
            <div><p className="eyebrow">Protected profile view</p><h2>{profile.name}</h2><p className="panel-note">Operational metadata only. Biometric embeddings stay in the local engine.</p></div>
            <button type="button" className="button-quiet" onClick={() => setSelectedId(null)}>Close profile</button>
          </div>
          {profileEdit && <div className="profile-edit-grid">
            <label><span>Name</span><input value={profileEdit.name} onChange={(event) => setProfileEdit((current) => ({ ...current, name: event.target.value }))} /></label>
            <label><span>Role</span><input value={profileEdit.role} onChange={(event) => setProfileEdit((current) => ({ ...current, role: event.target.value }))} /></label>
            <label><span>Class</span><input value={profileEdit.className} onChange={(event) => setProfileEdit((current) => ({ ...current, className: event.target.value }))} /></label>
            <label><span>Student ID</span><input value={profileEdit.studentId} onChange={(event) => setProfileEdit((current) => ({ ...current, studentId: event.target.value }))} /></label>
            <label className="profile-edit-wide"><span>Subjects, comma separated</span><input value={profileEdit.subjects} onChange={(event) => setProfileEdit((current) => ({ ...current, subjects: event.target.value }))} /></label>
            <button type="button" onClick={saveProfile} disabled={profileSaving}>{profileSaving ? <Loader2 className="spin" size={16} /> : <CheckCircle2 size={16} />} Save profile metadata</button>
          </div>}
          <div className="profile-detail-grid">
            <div><p className="eyebrow">Attendance history</p>{(profile.attendance_summary || []).slice(0, 8).map((record) => <div className="detail-row" key={record.id}><strong>{record.date}</strong><span>{record.clock_in || "No clock-in"} · {record.clock_out || "Open"}</span></div>)}{!profile.attendance_summary?.length && <p className="empty-copy">No attendance records.</p>}</div>
            <div><p className="eyebrow">Absence history</p>{(profile.absence_history || []).slice(0, 8).map((absence) => <div className="detail-row" key={absence.id}><strong>{absence.absence_date}</strong><span>{absence.status} · {absence.reason || "No reason"}</span></div>)}{!profile.absence_history?.length && <p className="empty-copy">No recorded absences.</p>}</div>
            <div><p className="eyebrow">Recognition evidence</p>{(profile.recognition_evidence || []).slice(0, 6).map((evidence) => <div className="detail-row" key={evidence.id}><strong>{evidence.decision}</strong><span>{evidence.identity_state || "UNRESOLVED"} · {Math.round(Number(evidence.quality_score || 0))} quality</span></div>)}{!profile.recognition_evidence?.length && <p className="empty-copy">No recognition evidence.</p>}</div>
            <div><p className="eyebrow">Related security events</p>{(profile.recent_events || []).slice(0, 6).map((event) => <div className="detail-row" key={event.id}><strong>{event.event_type}</strong><span>{event.timestamp}</span></div>)}{!profile.recent_events?.length && <p className="empty-copy">No related events.</p>}</div>
          </div>
        </>}
      </section>}
    </div>
  );
}
