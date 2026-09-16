import { getDemoState } from "./mockData";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const USE_DEMO_DATA = import.meta.env.VITE_USE_DEMO_DATA === "true";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function csrfToken() {
  return document.cookie.split(";").map((item) => item.trim()).find((item) => item.startsWith("optivox_csrf="))?.slice("optivox_csrf=".length) || "";
}

export const FRAME_URL = `${API_BASE}/api/live/frame`;

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`, { headers: requestHeaders(), credentials: "include" });
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return response.json();
}

export function requestHeaders(extra = {}) {
  const csrf = csrfToken();
  return { ...(csrf ? { "X-CSRF-Token": csrf } : {}), ...extra };
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
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      return { ...getEmptyState(), dataMode: "auth_required", authRequired: true };
    }
    const empty = getEmptyState();
    return { ...empty, dataMode: "backend_offline" };
  }
}

export async function login(username, password) {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return response.json();
}

export async function logout() {
  try {
    await fetch(`${API_BASE}/api/auth/logout`, {
      method: "POST",
      headers: requestHeaders(),
      credentials: "include",
    });
  } finally {
    // Session cookies are cleared by the backend response.
  }
}

export async function sendCommand(command, payload = {}) {
  try {
    return await fetch(`${API_BASE}/api/commands`, {
      method: "POST",
      headers: requestHeaders({ "Content-Type": "application/json", "Idempotency-Key": `web-${command}-${Date.now()}` }),
      credentials: "include",
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

export async function fetchEnrollmentStatus() {
  if (USE_DEMO_DATA) return { stage: "idle", message: "Demo mode does not control the local camera." };
  return getJson("/api/enrollment/status");
}

export async function fetchPerson(personId) {
  if (USE_DEMO_DATA) return getDemoState().people.find((person) => person.id === personId) || null;
  return getJson(`/api/people/${personId}`);
}

export async function updatePerson(personId, payload) {
  const response = await fetch(`${API_BASE}/api/people/${personId}`, {
    method: "PATCH",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function cancelEnrollment() {
  if (USE_DEMO_DATA) return { stage: "cancelled", message: "Demo enrollment cancelled." };
  return sendCommand("cancel_enrollment");
}

export async function confirmEnrollment(overrideDuplicate = false) {
  if (USE_DEMO_DATA) return { stage: "completed", message: "Demo enrollment confirmed." };
  return sendCommand("confirm_enrollment", { override_duplicate: Boolean(overrideDuplicate) });
}

export async function retrainPerson(name) {
  return sendCommand("retrain_person", { name });
}

export async function disablePerson(name) {
  return sendCommand("disable_person", { name });
}

export async function mergePeople(sourceName, targetName) {
  return sendCommand("merge_people", { source_name: sourceName, target_name: targetName });
}

export async function deletePerson(name) {
  return sendCommand("delete_person", { name, confirm: true });
}

export async function fetchOperationalSummary() {
  if (USE_DEMO_DATA) return {
    presenceSessions: 0,
    activePresenceSessions: 0,
    activeConfirmedEntities: 0,
    activeUnresolvedEntities: 0,
    recognitionEvidence: 0,
    confirmedEvidence: 0,
    rejectedEvidence: 0,
  };
  return getJson("/api/operations/summary");
}

export async function fetchAttendanceDecisions(limit = 40) {
  if (USE_DEMO_DATA) return [];
  return getJson(`/api/operations/attendance-decisions?limit=${Math.max(1, Math.min(Number(limit) || 40, 500))}`);
}

export async function fetchPerformanceReport() {
  if (USE_DEMO_DATA) return { measurement_status: "NOT_MEASURED" };
  return getJson("/api/system/performance");
}

export async function reviewEvent(eventId, action, note = "") {
  const response = await fetch(`${API_BASE}/api/events/${eventId}/review`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify({ action, note }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function reviewIncident(incidentId, action, note = "") {
  const response = await fetch(`${API_BASE}/api/incidents/${incidentId}/review`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify({ action, note }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchIncident(incidentId) {
  if (USE_DEMO_DATA) return null;
  return getJson(`/api/incidents/${incidentId}`);
}

export async function assignIncident(incidentId, assignee) {
  const response = await fetch(`${API_BASE}/api/incidents/${incidentId}/assign`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify({ assignee }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function recordAbsence(personId, payload) {
  const response = await fetch(`${API_BASE}/api/attendance/${personId}/absence`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function correctAttendance(personId, payload) {
  const response = await fetch(`${API_BASE}/api/attendance/${personId}/correct`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function clockInPerson(personId) {
  const response = await fetch(`${API_BASE}/api/attendance/${personId}/clock-in`, {
    method: "POST",
    headers: requestHeaders(),
    credentials: "include",
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return response.json();
}

export async function clockOutPerson(personId) {
  const response = await fetch(`${API_BASE}/api/attendance/${personId}/clock-out`, {
    method: "POST",
    headers: requestHeaders(),
    credentials: "include",
  });
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
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

export async function fetchCyberSummary() {
  if (USE_DEMO_DATA) return { incidentTotal: 0, openIncidents: 0, eventTotal: 0, statusCounts: [], categories: [], eventTypes: [], physicalEventsSeparate: true };
  return getJson("/api/cybersecurity/summary");
}

export async function fetchCyberIncidents(filters = {}) {
  if (USE_DEMO_DATA) return [];
  const params = new URLSearchParams();
  if (filters.status && filters.status !== "all") params.set("status", filters.status);
  if (filters.category && filters.category !== "all") params.set("category", filters.category);
  if (filters.severity && filters.severity !== "all") params.set("severity", filters.severity);
  params.set("limit", "100");
  return getJson(`/api/cybersecurity/incidents?${params.toString()}`);
}

export async function fetchCyberIncident(incidentId) {
  return getJson(`/api/cybersecurity/incidents/${incidentId}`);
}

export async function reviewCyberIncident(incidentId, action, note = "") {
  const response = await fetch(`${API_BASE}/api/cybersecurity/incidents/${incidentId}/review`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify({ action, note }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function assignCyberIncident(incidentId, assignee) {
  const response = await fetch(`${API_BASE}/api/cybersecurity/incidents/${incidentId}/assign`, {
    method: "POST",
    headers: requestHeaders({ "Content-Type": "application/json" }),
    credentials: "include",
    body: JSON.stringify({ assignee }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchCyberSessions() {
  if (USE_DEMO_DATA) return [];
  return getJson("/api/cybersecurity/sessions");
}

export async function fetchCyberAuthHistory() {
  if (USE_DEMO_DATA) return [];
  return getJson("/api/cybersecurity/auth-history?limit=100");
}

export async function fetchCyberAdminHistory() {
  if (USE_DEMO_DATA) return [];
  return getJson("/api/cybersecurity/admin-history?limit=100");
}

export async function fetchCyberIntegrityFailures() {
  if (USE_DEMO_DATA) return [];
  return getJson("/api/cybersecurity/integrity?limit=100");
}

export async function fetchCyberAlertFailures() {
  if (USE_DEMO_DATA) return [];
  return getJson("/api/cybersecurity/alert-failures?limit=100");
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
    operational: {
      presenceSessions: 0,
      activePresenceSessions: 0,
      activeConfirmedEntities: 0,
      activeUnresolvedEntities: 0,
      recognitionEvidence: 0,
      confirmedEvidence: 0,
      rejectedEvidence: 0,
    },
  };
}
