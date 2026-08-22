import { getDemoState } from "./mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const USE_DEMO_DATA = import.meta.env.VITE_USE_DEMO_DATA === "true";
const API_KEY = import.meta.env.VITE_OPTIVOX_API_KEY || "";

export const FRAME_URL = `${API_BASE}/api/live/frame`;

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`, { headers: requestHeaders() });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json();
}

function requestHeaders(extra = {}) {
  return { ...(API_KEY ? { "X-Optivox-Key": API_KEY } : {}), ...extra };
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body?.detail?.message || body?.detail || `Request failed: ${response.status}`;
  } catch {
    return `Request failed: ${response.status}`;
  }
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
      headers: requestHeaders({ "Content-Type": "application/json", "Idempotency-Key": `web-${command}-${Date.now()}` }),
      body: JSON.stringify({ command, payload }),
    }).then(async (response) => {
      if (!response.ok) throw new Error(await errorMessage(response));
      return response.json();
    });
  } catch (error) {
    if (error instanceof Error && !error.message.includes("Failed to fetch")) throw error;
    throw new Error("Backend command endpoint is unavailable or rejected the command.");
  }
}

export async function reviewEvent(eventId, action, note = "") {
  const response = await fetch(`${API_BASE}/api/events/${eventId}/review`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ action, note }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function reviewIncident(incidentId, action, note = "") {
  const response = await fetch(`${API_BASE}/api/incidents/${incidentId}/review`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ action, note }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function correctAttendance(personId, payload) {
  const response = await fetch(`${API_BASE}/api/attendance/${personId}/correct`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchAttendanceCalendar(year, month) {
  if (USE_DEMO_DATA) return getDemoCalendar(year, month);
  return getJson(`/api/attendance/calendar?year=${year}&month=${month}`);
}

export async function fetchAcademicOverview(year, month) {
  if (USE_DEMO_DATA) return getDemoAcademic(year, month);
  return getJson(`/api/academic/overview?year=${year}&month=${month}`);
}

export async function fetchAnalytics() {
  if (USE_DEMO_DATA) return getDemoAnalytics();
  const [attendance, security, objects] = await Promise.all([
    getJson("/api/analytics/attendance?days=30"),
    getJson("/api/analytics/security"),
    getJson("/api/analytics/objects"),
  ]);
  return { ...attendance, ...security, ...objects };
}

function getDemoCalendar(year, month) {
  const days = Array.from({ length: new Date(year, month, 0).getDate() }, (_, index) => `${year}-${String(month).padStart(2, "0")}-${String(index + 1).padStart(2, "0")}`);
  return { year, month, days, people: peopleForCalendar(days) };
}

function peopleForCalendar(days) {
  return getDemoState().people.map((person) => ({
    id: person.id,
    name: person.name,
    role: person.role,
    subjects: person.subjects || [],
    records: Object.fromEntries(days.map((day, index) => [day, index % 7 === person.id % 5 ? null : { status: person.status === "Not Yet Detected" ? "Not Yet Detected" : index % 9 === 0 ? "Late" : "Present", clockIn: person.clockIn, clockOut: person.clockOut }]))
  }));
}

function getDemoAcademic(year, month) {
  return { year, month, subjects: ["Computer Vision", "Mathematics", "Science"], profiles: getDemoState().people.map((person) => ({ person_id: person.id, name: person.name, subjects: person.subjects || ["Computer Vision"] })), absence_records: [], absence_note: "Demo absence data is illustrative." };
}

function getDemoAnalytics() {
  const demo = getDemoState();
  return { ...demo.analytics, attendanceByDay: demo.analytics.attendanceByDay, byStatus: [{ name: "Present", value: 4 }, { name: "Late", value: 1 }, { name: "Not detected", value: 1 }], methodSplit: [{ name: "Automatic", value: 4 }, { name: "Manual", value: 1 }], rosterTotals: { registered: demo.people.length, active: demo.people.filter((person) => person.active).length, seenToday: demo.people.filter((person) => person.status !== "Not Yet Detected").length } };
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
      securityObservations: 0,
      openIncidents: 0,
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
    incidents: [],
  };
}
