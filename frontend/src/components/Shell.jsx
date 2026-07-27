import {
  Activity,
  BarChart3,
  Camera,
  ClipboardCheck,
  Database,
  Radio,
  ShieldAlert,
  Users,
} from "lucide-react";
import clsx from "../utils/clsx";
import { StatusBadge } from "./StatusBadge";

const navItems = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "attendance", label: "Attendance", icon: ClipboardCheck },
  { id: "security", label: "Security", icon: ShieldAlert },
  { id: "people", label: "People", icon: Users },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "system", label: "System", icon: Database },
];

export default function Shell({ activePage, onNavigate, connection, state, children }) {
  const engine = state?.engine;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">O</div>
          <div>
            <strong>Optivox</strong>
            <span>Command Center</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={clsx("nav-item", activePage === item.id && "active")}
                key={item.id}
                onClick={() => onNavigate(item.id)}
                type="button"
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-status">
          <div className="status-row">
            <Radio size={16} />
            <span>{connectionLabel(connection)}</span>
          </div>
          <div className="status-row">
            <Camera size={16} />
            <span>{engine?.camera || "Waiting"}</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local exhibition system</p>
            <h1>{pageTitle(activePage)}</h1>
          </div>
          <div className="topbar-actions">
            <StatusBadge value={engine?.mode || "Local AI Processing"} tone="info" />
            <StatusBadge value={connectionLabel(connection)} tone={connection === "live" ? "success" : connection === "demo" ? "info" : "warning"} />
            <span className="clock">{state?.localTime || "--:--:--"}</span>
          </div>
        </header>

        {connection !== "live" && (
          <div className="banner">
            {offlineMessage(connection)}
          </div>
        )}

        {children}
      </main>
    </div>
  );
}

function connectionLabel(connection) {
  return {
    live: "LIVE DATA",
    demo: "DEMO MODE",
    backend_offline: "BACKEND OFFLINE",
    engine_offline: "ENGINE OFFLINE",
    connecting: "CONNECTING",
  }[connection] || "CONNECTING";
}

function offlineMessage(connection) {
  return {
    demo: "Demo mode is explicitly enabled. The interface is showing exhibition sample data.",
    backend_offline: "The FastAPI backend is unavailable. No fake detections are being shown in live mode.",
    engine_offline: "The backend is online, but the vision engine is not publishing a fresh heartbeat.",
    connecting: "Connecting to the local OptiVox backend.",
  }[connection] || "Live data is temporarily unavailable.";
}

function pageTitle(page) {
  return {
    overview: "Live Monitor",
    attendance: "Attendance",
    security: "Security Events",
    people: "People and Enrollment",
    analytics: "Analytics",
    system: "System Status",
  }[page] || "Optivox";
}
