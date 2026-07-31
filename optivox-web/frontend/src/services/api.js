const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export function apiUrl(endpoint) {
  const normalized = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${normalized}`;
}

export async function apiRequest(endpoint, options = {}) {
  const response = await fetch(apiUrl(endpoint), {
    ...options,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const data = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    throw new Error(data?.message || data?.detail || `Request failed with status ${response.status}`);
  }

  return response.status === 204 ? null : data;
}
