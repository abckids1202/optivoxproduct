import { BellRing, Camera, Cpu, Database, HardDrive, Radio } from "lucide-react";
import StatCard from "../components/StatCard";
import { sendCommand } from "../services/api";

export default function System({ state, connection }) {
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
        <StatCard label="Bridge" value={connection === "live" ? "Connected" : "Demo"} detail="FastAPI ready path" icon={Radio} tone="info" />
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
            ["Face detector", "Active"],
            ["Face recognizer", "Active"],
            ["Object detector", "Active"],
            ["Pose detector", "Active"],
            ["Anti-spoofing", "Active"],
            ["Danger model", "Optional"],
            ["SQLite database", "Connected"],
            ["Alert channels", "Configured"],
          ].map(([name, status]) => (
            <div className="system-row" key={name}>
              <span>{name}</span>
              <strong>{status}</strong>
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
