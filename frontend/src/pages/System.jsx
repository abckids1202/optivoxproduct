import { BellRing, Camera, Cpu, Database, Gauge, HardDrive, Radio } from "lucide-react";
import { useEffect, useState } from "react";
import StatCard from "../components/StatCard";
import { fetchAttendanceDecisions, fetchLivenessChallenges, fetchPerformanceReport, fetchSecurityZones, fetchSyncStatus, sendCommand } from "../services/api";

export default function System({ state, connection }) {
  const [performanceReport, setPerformanceReport] = useState(null);
  const [attendanceDecisions, setAttendanceDecisions] = useState([]);
  const [livenessChallenges, setLivenessChallenges] = useState([]);
  const [securityZones, setSecurityZones] = useState({ source: "startup_config", zones: [] });
  const [syncStatus, setSyncStatus] = useState(null);
  const runtime = state.runtime || {};
  const capabilities = runtime.capabilities || {};
  const performance = state.performance || {};
  const latency = performance.latency_ms || {};
  const counters = performance.counters || {};
  const visionPerformance = performance.vision || {};
  const visionCounters = visionPerformance;
  const models = visionPerformance.models || {};
  const modelAdapters = visionPerformance.model_adapters?.models || [];
  const resources = performance.resource || {};
  const operational = state.operational || {};
  const centerAttendance = state.attendance?.center_mode || {};
  const edgeSync = runtime.edge_sync || {};

  useEffect(() => {
    Promise.all([fetchPerformanceReport(), fetchAttendanceDecisions(), fetchLivenessChallenges(), fetchSecurityZones(), fetchSyncStatus()])
      .then(([report, decisions, challenges, zones, sync]) => {
        setPerformanceReport(report);
        setAttendanceDecisions(decisions || []);
        setLivenessChallenges(challenges || []);
        setSecurityZones(zones || { source: "unknown", zones: [] });
        setSyncStatus(sync || null);
      })
      .catch(() => {
        setPerformanceReport(null);
        setAttendanceDecisions([]);
        setLivenessChallenges([]);
        setSecurityZones({ source: "unavailable", zones: [] });
        setSyncStatus(null);
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
            ["Security policy configuration", capabilities.security_configuration],
            ["Visual intrusion zones", capabilities.security_intrusion],
            ["Loitering rules", capabilities.security_loitering],
            ["Running detection", capabilities.security_running],
            ["Evacuation signal", capabilities.security_evacuation],
            ["PPE compliance", capabilities.security_ppe],
            ["Vehicle tracking", capabilities.vehicle_tracking],
            ["Calibrated vehicle speed", capabilities.vehicle_speed],
            ["Plate reading", capabilities.plate_reading],
            ["Edge synchronization", capabilities.edge_sync],
            ["Model registry", capabilities.model_registry],
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
            <p className="eyebrow">Adapter health</p>
            <h2>Perception model pipeline</h2>
          </div>
          <span className="status-badge tone-info">{modelAdapters.length} components</span>
        </div>
        {modelAdapters.length ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Component</th><th>Task</th><th>State</th><th>Mode</th><th>Promotion</th><th>Calls</th><th>Failures</th><th>Latency</th></tr></thead>
              <tbody>{modelAdapters.map((model) => {
                const state = String(model.state || "NOT_MEASURED");
                const tone = state === "AVAILABLE" ? "tone-success" : state === "FAILED_RUNTIME" || state === "FAILED_INTEGRITY" ? "tone-danger" : "tone-warning";
                return (
                  <tr key={model.name}>
                    <td>{model.name}</td>
                    <td>{model.task}</td>
                    <td><span className={`status-badge ${tone}`}>{state}</span></td>
                    <td>{model.execution_mode || "-"}</td>
                    <td>{model.promotion_status || "not applicable"}</td>
                    <td>{model.calls ?? 0}</td>
                    <td>{model.failures ?? 0}</td>
                    <td>{model.latency?.p95_ms == null ? "Not measured" : `${model.latency.p95_ms} ms p95`}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
        ) : <p className="empty-copy">Model adapter telemetry is not available yet. Start the local engine to publish it.</p>}
        <p className="panel-note">Capability state is diagnostic metadata. Models produce observations; CorrelationCore and policy rules remain authoritative for attendance and security.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Intrusion policy</p>
            <h2>Restricted zones</h2>
          </div>
          <span className={`status-badge ${securityZones.zones?.length ? "tone-success" : "tone-warning"}`}>
            {securityZones.zones?.length ? `${securityZones.zones.length} configured` : "Not configured"}
          </span>
        </div>
        {securityZones.zones?.length ? (
          <div className="system-grid">
            {securityZones.zones.map((zone) => (
              <div className="system-row" key={zone.id}>
                <span>{zone.name || zone.id} · {zone.camera_id || "all cameras"}</span>
                <strong>{zone.restricted === false ? "Monitor" : `Restricted · severity ${zone.severity ?? 0}`}</strong>
              </div>
            ))}
          </div>
        ) : <p className="empty-copy">No restricted zones are active. Intrusion detection remains unavailable until a validated zone policy is installed.</p>}
        <p className="panel-note">Policy source: <strong>{securityZones.source}</strong>. Zone geometry and access rules are stored locally; sensitive biometric data is not exposed here.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Liveness evidence</p>
            <h2>Entity-scoped challenge trail</h2>
          </div>
          <span className="status-badge tone-info">{livenessChallenges.length} recent challenges</span>
        </div>
        {livenessChallenges.length ? (
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>State</th><th>Phase</th><th>Entity</th><th>Session</th><th>Status</th><th>Attempt</th><th>Updated</th><th>Reason</th></tr></thead>
              <tbody>{livenessChallenges.map((challenge) => (
                <tr key={challenge.id}>
                  <td><span className={`status-badge ${challenge.challengeState === "PASSED" ? "tone-success" : challenge.challengeState === "IN_PROGRESS" ? "tone-info" : "tone-warning"}`}>{challenge.challengeState}</span></td>
                  <td>{challenge.phase}</td>
                  <td>{challenge.entityId || "-"}</td>
                  <td>{challenge.presenceSessionId ?? "-"}</td>
                  <td>{challenge.livenessStatus}</td>
                  <td>{challenge.attemptNumber}</td>
                  <td>{challenge.updatedAt || "-"}</td>
                  <td>{challenge.failureReason || "-"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <p className="empty-copy">No liveness challenges have been persisted yet.</p>}
        <p className="panel-note">This trace records challenge state and timing only. Raw face images, embeddings, and biometric templates are never returned here.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Automatic attendance</p>
            <h2>Center verification gate</h2>
          </div>
          <span className={`status-badge ${centerAttendance.enabled ? "tone-info" : "tone-neutral"}`}>
            {centerAttendance.enabled ? "ACTIVE" : "OFF"}
          </span>
        </div>
        <div className="system-grid">
          {[
            ["Gate phase", centerAttendance.phase || "READY"],
            ["Candidate", centerAttendance.candidate || "Waiting for one person"],
            ["Progress", centerAttendance.progress == null ? "Not started" : `${Math.round(Number(centerAttendance.progress) * 100)}%`],
            ["Liveness phase", centerAttendance.liveness_phase || "CENTER"],
            ["Liveness status", centerAttendance.liveness_status || "NOT_EVALUATED"],
            ["Challenge attempts", centerAttendance.liveness_attempts ?? 0],
          ].map(([name, value]) => <div className="system-row" key={name}><span>{name}</span><strong>{value}</strong></div>)}
        </div>
        <p className="panel-note">{centerAttendance.status || "Center attendance is controlled by the local edge runtime."} Automatic clock-in still requires correlated identity, quality, presence, and liveness gates.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Edge boundary</p>
            <h2>Minimized synchronization queue</h2>
          </div>
          <Radio size={20} aria-hidden="true" />
        </div>
        <div className="system-grid">
          {[
            ["Security mode", edgeSync.security_mode || "Not started"],
            ["Queue integrity", edgeSync.integrity || "Not measured"],
            ["Pending records", edgeSync.pending == null ? "Not measured" : edgeSync.pending],
            ["Queued records", edgeSync.records == null ? "Not measured" : edgeSync.records],
          ].map(([name, value]) => <div className="system-row" key={name}><span>{name}</span><strong>{value}</strong></div>)}
        </div>
        <p className="panel-note">The edge queue contains minimized operational provenance only. Raw frames, embeddings, credentials, and local plate text stay on the device.</p>
      </section>

      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Control-plane receipt</p>
            <h2>Synchronization checkpoint</h2>
          </div>
          <span className={`status-badge ${syncStatus?.checkpoint?.status === "active" ? "tone-success" : "tone-warning"}`}>
            {syncStatus?.checkpoint?.status || "Not connected"}
          </span>
        </div>
        <div className="system-grid">
          {[
            ["Receiver configured", syncStatus?.receiver_configured ? "Yes" : "No"],
            ["Last accepted sequence", syncStatus?.checkpoint?.sequence ?? "Not measured"],
            ["Batches received", syncStatus?.batches_received ?? "Not measured"],
            ["Events received", syncStatus?.events_received ?? "Not measured"],
            ["Last received", syncStatus?.checkpoint?.last_received_at || "Never"],
          ].map(([name, value]) => <div className="system-row" key={name}><span>{name}</span><strong>{value}</strong></div>)}
        </div>
        {syncStatus?.devices?.length > 1 && (
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Device</th><th>Site</th><th>Organization</th><th>Checkpoint</th><th>Last receipt</th><th>Events</th></tr></thead>
              <tbody>{syncStatus.devices.map((device) => (
                <tr key={`${device.organization_id}:${device.site_id}:${device.device_id}`}>
                  <td>{device.device_id}</td>
                  <td>{device.site_id}</td>
                  <td>{device.organization_id}</td>
                  <td>{device.checkpoint?.sequence ?? 0} · {device.checkpoint?.status || "never_connected"}</td>
                  <td>{device.checkpoint?.last_received_at || "Never"}</td>
                  <td>{device.events_received ?? 0}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
        <p className="panel-note">This is the receiver checkpoint, not a frame counter. A device can continue operating locally while the checkpoint remains unchanged; the queue integrity and pending count above show what is waiting at the edge.</p>
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
            ["Liveness challenges", operational.livenessChallenges],
            ["Liveness passed", operational.passedLivenessChallenges],
            ["Liveness failed or timed out", operational.failedLivenessChallenges],
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
