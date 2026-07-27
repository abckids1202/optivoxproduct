import { getDemoState } from "./mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const USE_DEMO_DATA = import.meta.env.VITE_USE_DEMO_DATA === "true";

export const FRAME_URL = `${API_BASE}/api/live/frame`;

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchDashboardState() {
  if (USE_DEMO_DATA) {
    return { ...getDemoState(), dataMode: "demo" };
  }
  try {
    const state = await getJson("/api/live/status");
    return { ...state, dataMode: "live" };
  } catch {
    const empty = getEmptyState();
    return { ...empty, dataMode: "backend_offline" };
  }
}

export async function sendCommand(command, payload = {}) {
  try {
    return await fetch(`${API_BASE}/api/commands`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, payload }),
    });
  } catch {
    throw new Error("Backend command endpoint is unavailable.");
  }
}

function getEmptyState() {
  const now = new Date();
  return {
    generatedAt: now.toISOString(),
    localTime: now.toLocaleTimeString("en-GB"),
    engine: {
      status: "Offline",
      camera: "Unknown",
      location: "Class",
      fps: 0,
      uptime: "0s",
      mode: "Local AI Processing",
      frameAge: "unavailable",
      frameAvailable: false,
    },
    summary: {
      presentToday: 0,
      visibleNow: 0,
      unknownToday: 0,
      securityEvents: 0,
      alertsSent: 0,
      registeredPeople: 0,
      present: 0,
      late: 0,
      left: 0,
      not_yet_detected: 0,
    },
    visiblePeople: [],
    objects: [],
    people: [],
    attendance: [],
    events: [],
    analytics: {
      attendanceByDay: [],
      eventsByHour: [],
      securityCategories: [],
    },
  };
}
