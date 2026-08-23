import { BellRing, Camera, Cpu, Database, Gauge, HardDrive, Radio } from "lucide-react";
import StatCard from "../components/StatCard";
import { sendCommand } from "../services/api";

export default function System({ state, connection }) {
  const runtime = state.runtime || {};
  const capabilities = runtime.capabilities || {};
  const performance = state.performance || {};
  const latency = performance.latency_ms || {};
  const counters = performance.counters || {};
  const visionPerformance = performance.vision || {};
  const visionCounters = visionPerformance;

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
            <p className="eyebrow">Performance telemetry</p>
            <h2>Live pipeline health</h2>
          </div>
          <Gauge size={20} aria-hidden="true" />
        </div>
        <div className="system-grid">
          {[
            ["Inference rate", `${performance.inference_fps || 0} FPS`],
            ["Inference p95", `${latency.inference_p95 || 0} ms`],
            ["Latest frame age", `${latency.latest_frame_age || 0} ms`],
            ["Stale frames dropped", counters.stale_frames_dropped || 0],
            ["Side-effect queue", performance.queue_depths?.side_effects || 0],
            ["Critical queue drops", counters.critical_queue_drops || 0],
            ["Recognition cache hits", visionCounters.recognition_cache_hits || 0],
            ["ROI inference runs", visionCounters.roi_inference_runs || 0],
          ].map(([name, value]) => (
            <div className="system-row" key={name}>
              <span>{name}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
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
