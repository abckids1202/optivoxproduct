import { BellRing, Camera, Cpu, Database, Gauge, HardDrive, Radio } from "lucide-react";
import { useEffect, useState } from "react";
import StatCard from "../components/StatCard";
import { fetchAttendanceDecisions, fetchPerformanceReport, sendCommand } from "../services/api";

export default function System({ state, connection }) {
  const [performanceReport, setPerformanceReport] = useState(null);
  const [attendanceDecisions, setAttendanceDecisions] = useState([]);
  const runtime = state.runtime || {};
  const capabilities = runtime.capabilities || {};
  const performance = state.performance || {};
  const latency = performance.latency_ms || {};
  const counters = performance.counters || {};
  const visionPerformance = performance.vision || {};
  const visionCounters = visionPerformance;
  const models = visionPerformance.models || {};
  const resources = performance.resource || {};
  const operational = state.operational || {};

  useEffect(() => {
    Promise.all([fetchPerformanceReport(), fetchAttendanceDecisions()])
      .then(([report, decisions]) => {
        setPerformanceReport(report);
        setAttendanceDecisions(decisions || []);
      })
      .catch(() => {
        setPerformanceReport(null);
        setAttendanceDecisions([]);
      });
  }, [state.generatedAt]);

  function metric(value, suffix = "") {
    return value === null || value === undefined ? "Not measured" : `${value}${suffix}`;
  }

  async function runCommand(command, confirmText) {
    if (confirmText && !window.confirm(confirmText)) return;
    try {
      await sendCommand(command);
      window.alert("Command queued for the local OptiVox engine.");
    } catch (error) {
      window.alert(error.message);
    }
  }

  return (
    <div className="page-stack">
      <div className="stats-grid four">
        <StatCard label="Engine" value={state.engine.status} detail={state.engine.uptime} icon={Cpu} tone="success" />
        <StatCard label="Camera" value={state.engine.camera} detail={state.engine.location} icon={Camera} tone="success" />
        <StatCard label="Bridge" value={connection === "live" ? "Connected" : "Offline"} detail={runtime.version || "Runtime not reported"} icon={Radio} tone="info" />
        <StatCard label="Storage" value="Local" detail="SQLite and snapshots" icon={HardDrive} tone="neutral" />
      </div>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Model status</p>
            <h2>Local engine capabilities</h2>
          </div>
        </div>
        <div className="system-grid">
          {[
            ["Face detector", capabilities.face_detection],
            ["Face recognizer", capabilities.face_recognition],
            ["Object detector", capabilities.object_detection],
            ["Pose detector", capabilities.pose_detection],
            ["Guided liveness", capabilities.guided_liveness],
            ["Visual intrusion zones", capabilities.security_intrusion],
            ["Loitering rules", capabilities.security_loitering],
            ["Running detection", capabilities.security_running],
            ["Evacuation signal", capabilities.security_evacuation],
            ["PPE compliance", capabilities.security_ppe],
            ["ML anti-spoof model", capabilities.ml_anti_spoof],
            ["Web enrollment", capabilities.web_enrollment],
            ["Runtime bridge", capabilities.runtime_bridge],
            ["SQLite database", "local"],
          ].map(([name, status]) => (
            <div className="system-row" key={name}>
              <span>{name}</span>
              <strong>{typeof status === "boolean" ? (status ? "Available" : "Unavailable") : status}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Attendance decision trail</p>
            <h2>Why the gate accepted or rejected a face</h2>
          </div>
          <span className="status-badge tone-info">{attendanceDecisions.length} recent decisions</span>
        </div>
        {attendanceDecisions.length ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Decision</th><th>Identity</th><th>Liveness</th><th>Quality</th><th>Confidence</th><th>Reason</th><th>Frame</th></tr></thead>
              <tbody>{attendanceDecisions.map((decision) => (
                <tr key={decision.id}>
                  <td><span className={`status-badge ${decision.decision === "eligible" ? "tone-success" : "tone-warning"}`}>{decision.decision}</span></td>
                  <td>{decision.identityState || "UNRESOLVED"}</td>
                  <td>{decision.livenessStatus || "NOT_EVALUATED"}</td>
                  <td>{decision.qualityScore == null ? "Not measured" : Math.round(Number(decision.qualityScore))}</td>
                  <td>{decision.recognitionConfidence == null ? "Not measured" : Number(decision.recognitionConfidence).toFixed(3)}</td>
                  <td>{decision.reason || "-"}</td>
                  <td>{decision.sourceFrameId ?? "-"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <p className="empty-copy">No automatic gate decisions have been recorded yet.</p>}
        <p className="panel-note">A decision is evidence of the automatic gate, not itself an attendance record. Unknown, uncertain, and spoof-suspect decisions remain rejected.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Correlation core</p>
            <h2>Operational state</h2>
          </div>
          <Radio size={20} aria-hidden="true" />
        </div>
        <div className="system-grid">
          {[
            ["Active presence sessions", operational.activePresenceSessions],
            ["Confirmed entities", operational.activeConfirmedEntities],
            ["Unresolved entities", operational.activeUnresolvedEntities],
            ["Recognition evidence", operational.recognitionEvidence],
            ["Confirmed evidence", operational.confirmedEvidence],
            ["Rejected or uncertain evidence", operational.rejectedEvidence],
          ].map(([name, value]) => <div className="system-row" key={name}><span>{name}</span><strong>{metric(value)}</strong></div>)}
        </div>
        <p className="panel-note">Sessions represent correlated presence. Evidence represents recognition decisions. Neither count is a frame total.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Performance telemetry</p>
            <h2>Live pipeline health</h2>
          </div>
          <Gauge size={20} aria-hidden="true" />
        </div>
        <div className="system-grid">
          {[
            ["Inference rate", metric(performance.inference_fps, " FPS")],
            ["Capture rate", metric(performance.capture_fps, " FPS")],
            ["Display rate", metric(performance.display_fps, " FPS")],
            ["Inference p95", metric(latency.inference_p95, " ms")],
            ["Capture-to-display p95", metric(latency.end_to_end_p95, " ms")],
            ["Frame age at start", metric(latency.frame_age_start_avg, " ms avg")],
            ["Frame age at completion", metric(latency.frame_age_end_avg, " ms avg")],
            ["Frame age at display", metric(latency.display_frame_age_avg, " ms avg")],
            ["Latest frame age", metric(latency.latest_frame_age, " ms")],
            ["Frames consumed", metric(counters.frames_consumed)],
            ["Frames replaced", metric(counters.frames_replaced)],
            ["Frame IDs skipped", metric(counters.frame_ids_skipped)],
            ["Stale frames dropped", metric(counters.stale_frames_dropped)],
            ["Side-effect queue", metric(performance.queue_depths?.side_effects)],
            ["Critical queue drops", metric(counters.critical_queue_drops)],
            ["Recognition cache hits", metric(visionCounters.recognition_cache_hits)],
            ["Embedding skips", metric(visionCounters.embeddings_skipped_due_to_cache)],
            ["Embedding calls/sec", metric(models.face_embedding?.calls_per_second)],
            ["Matcher calls/sec", metric(models.identity_matching?.calls_per_second)],
            ["ROI inference runs", metric(visionCounters.roi_inference_runs)],
            ["Process CPU", resources.cpu_percent == null ? "Not measured" : `${resources.cpu_percent}%`],
            ["Process RAM", resources.process_rss_mb == null ? "Not measured" : `${resources.process_rss_mb} MB`],
          ].map(([name, value]) => (
            <div className="system-row" key={name}>
              <span>{name}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
        {performanceReport && <p className="panel-note">Saved benchmark: <strong>{performanceReport.measurement_status}</strong>{performanceReport.benchmark_age_seconds != null ? ` · ${Math.round(performanceReport.benchmark_age_seconds / 3600)}h old` : ""}. GPU and VRAM remain explicitly unmeasured when the runtime cannot provide them.</p>}
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Exhibition controls</p>
            <h2>Safe actions</h2>
          </div>
        </div>
        <div className="control-row">
          <button type="button" onClick={() => runCommand("test_alert")}><BellRing size={16} /> Test alert</button>
          <button type="button" onClick={() => runCommand("save_snapshot")}><Camera size={16} /> Save snapshot</button>
          <button type="button" onClick={() => runCommand("reset_demo_data", "This only queues a safe non-destructive demo reset command. Continue?")}><Database size={16} /> Demo reset</button>
        </div>
      </section>
    </div>
  );
}
