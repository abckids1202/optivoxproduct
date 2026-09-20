export const branchMeta = {
  core: { label: "OptiVox Core", color: "#d9b7ff" },
  perception: { label: "Perception", color: "#65d8ff" },
  identity: { label: "Identity", color: "#b98cff" },
  attendance: { label: "Attendance", color: "#56e0aa" },
  security: { label: "Security", color: "#ffb547" },
  data: { label: "Data & Memory", color: "#8fc7ff" },
  response: { label: "Response", color: "#ff8a9a" },
  interface: { label: "Interface", color: "#b8c8d9" },
};

const n = (id, branch, kind, title, short, status, tech, x, y) => ({
  id, branch, kind, title, short, status, tech, x, y,
  purpose: short,
  input: "Local camera frames, identity data, or configured events",
  process: "OptiVox evaluates the signal with configured rules and local processing.",
  output: "Structured context for the next stage of the system",
  limitation: "Results depend on camera quality, configuration, and human review.",
});

export const mapNodes = [
  n("core", "core", "core", "OptiVox Core", "Coordinates perception, memory, and action.", "Core Feature", "Python orchestration layer", 50, 48),
  n("camera", "perception", "primary", "Camera Input", "Receives a local video stream.", "Core Feature", "OpenCV", 13, 20),
  n("face-detection", "perception", "feature", "Face Detection", "Locates faces and visual landmarks.", "Core Feature", "InsightFace / OpenCV", 29, 10),
  n("object-detection", "perception", "feature", "Object Detection", "Recognizes objects and event classes.", "Working Prototype", "YOLO models", 26, 67),
  n("pose-analysis", "perception", "feature", "Pose Analysis", "Reads body geometry over time.", "Experimental", "MediaPipe", 17, 83),
  n("face-embedding", "identity", "feature", "Face Embedding", "Turns a face into a numerical representation.", "Core Feature", "InsightFace embeddings", 40, 17),
  n("similarity-search", "identity", "feature", "Similarity Search", "Compares a new face with enrolled samples.", "Core Feature", "FAISS index", 53, 12),
  n("confirmation", "identity", "feature", "Temporal Confirmation", "Requires repeated recognition before acting.", "Core Feature", "Frame-window smoothing", 62, 26),
  n("known", "identity", "feature", "Known Identity", "A confirmed match to an enrolled person.", "Core Feature", "Recognition pipeline", 72, 18),
  n("unknown", "security", "feature", "Unknown Presence", "A temporary identity for an unresolved match.", "Working Prototype", "Event tracker", 70, 47),
  n("clock-in", "attendance", "feature", "Automatic Clock-In", "Turns a confirmed identity into a record.", "Core Feature", "Attendance manager / SQLite", 82, 32),
  n("attendance-db", "data", "feature", "Attendance Records", "Keeps the history of attendance events.", "Core Feature", "SQLite", 91, 52),
  n("security-event", "security", "feature", "Security Event", "Records an observable event for review.", "Working Prototype", "Event manager", 77, 63),
  n("snapshot", "data", "feature", "Snapshot", "Preserves the relevant frame for context.", "Working Prototype", "OpenCV / local storage", 87, 72),
  n("alert", "response", "feature", "Alert Channels", "Routes important events to configured destinations.", "Working Prototype", "Alert manager", 86, 86),
  n("dashboard", "interface", "primary", "Dashboard", "Turns system output into an operational view.", "Core Feature", "React / Flask API", 67, 78),
  n("reports", "interface", "feature", "Reports", "Summarizes what the system has remembered.", "Working Prototype", "React / SQLite", 55, 91),
];

export const mapEdges = [
  ["camera", "face-detection", "perception"], ["camera", "object-detection", "perception"], ["camera", "pose-analysis", "perception"],
  ["face-detection", "face-embedding", "identity"], ["face-embedding", "similarity-search", "identity"], ["similarity-search", "confirmation", "identity"], ["confirmation", "known", "identity"], ["confirmation", "unknown", "security"],
  ["known", "clock-in", "attendance"], ["clock-in", "attendance-db", "data"], ["attendance-db", "dashboard", "interface"], ["attendance-db", "reports", "interface"],
  ["unknown", "security-event", "security"], ["object-detection", "security-event", "security"], ["pose-analysis", "security-event", "security"], ["security-event", "snapshot", "data"], ["security-event", "alert", "response"], ["snapshot", "dashboard", "interface"], ["alert", "dashboard", "interface"],
  ["core", "camera", "core"], ["core", "dashboard", "core"],
];

export const flowPresets = [
  { id: "attendance-flow", label: "How attendance works", color: "#56e0aa", nodes: ["camera", "face-detection", "face-embedding", "similarity-search", "confirmation", "known", "clock-in", "attendance-db", "dashboard", "reports"] },
  { id: "unknown-flow", label: "How unknown presence works", color: "#ffb547", nodes: ["camera", "face-detection", "face-embedding", "similarity-search", "confirmation", "unknown", "security-event", "snapshot", "dashboard"] },
  { id: "security-flow", label: "How alerts work", color: "#ff8a9a", nodes: ["camera", "object-detection", "pose-analysis", "security-event", "snapshot", "alert", "dashboard"] },
  { id: "memory-flow", label: "How data is stored", color: "#8fc7ff", nodes: ["known", "clock-in", "attendance-db", "security-event", "snapshot", "reports", "dashboard"] },
];

export const guidedSequence = [
  { nodeId: "camera", title: "1 / Observation", narration: "OptiVox receives a local camera stream and prepares frames for analysis." },
  { nodeId: "face-detection", title: "2 / Perception", narration: "The system locates faces and other observable signals before asking what they mean." },
  { nodeId: "face-embedding", title: "3 / Recognition", narration: "A detected face becomes a representation that can be compared with enrolled samples." },
  { nodeId: "confirmation", title: "4 / Decision", narration: "Repeated recognition creates stronger evidence before the system acts." },
  { nodeId: "clock-in", title: "5 / Attendance", narration: "A confirmed identity can become an attendance record." },
  { nodeId: "unknown", title: "6 / Security", narration: "An unresolved identity is preserved as context for human review." },
  { nodeId: "dashboard", title: "7 / Meaning", narration: "Perception becomes operational when it can be reviewed and understood." },
];
