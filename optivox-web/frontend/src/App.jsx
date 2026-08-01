import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { apiRequest, apiUrl } from "./services/api";
import IntelligenceMap from "./components/IntelligenceMap";
import FeatureNavigator from "./components/FeatureNavigator";
import CinematicHero from "./components/CinematicHero";
import { featureContent } from "./content/features";

const year = new Date().getFullYear();

const capabilities = [
  ["Face Recognition", "Locally compares enrolled identity profiles after repeated confirmation across frames.", "IDENTITY"],
  ["Automated Attendance", "Records attendance after a confirmed enrolled identity and helps prevent duplicate entries.", "RECORDS"],
  ["Unknown-Person Monitoring", "Tracks unresolved presence as context without treating unknown as dangerous.", "CONTEXT"],
  ["Anti-Spoofing", "Experimental liveness checks look for observable facial movement before trusting a match.", "EXPERIMENTAL"],
  ["Object and Safety Events", "Selected objects, pose signals, and congestion rules can become reviewable events.", "OBSERVATION"],
  ["Local Data and Reports", "Attendance, events, profiles, alerts, and snapshots stay organised in local storage.", "MEMORY"],
  ["Dashboard and Assistant", "A web interface turns structured runtime data into operational context.", "INTERFACE"],
  ["Alerts and Evidence", "Configured channels can receive event notifications with relevant snapshots.", "RESPONSE"],
];

const featureRows = [
  {
    name: "Face Recognition & Enrollment",
    desc: "Faces are encoded into 512-dimensional embeddings and matched with FAISS for fast lookups. Thresholds can be tuned per person from enrollment variance.",
    points: ["Multi-angle passive enrollment wizard", "Per-identity threshold tuning", "Majority-vote smoothing across frames", "Image-quality gating before enrollment"],
    label: "FACE_ANALYZER - FAISSIndexer",
    boxes: [["Known - 0.94", "28%", "24%", "24%", "44%", "var(--ok)"]],
  },
  {
    name: "Anti-Spoofing & Liveness",
    desc: "Before a face is trusted, OptiVox checks whether it is a living person using blink detection, head-pose variance, and depth consistency.",
    points: ["EAR-based blink detection", "Head-pose movement variance", "Depth-consistency estimation", "REAL / UNCERTAIN / SUSPECT states"],
    label: "AntiSpoofDetector",
    boxes: [["LIVE - REAL", "30%", "30%", "30%", "42%", "var(--ok)"]],
  },
  {
    name: "Weapon, Fire & Smoke Detection",
    desc: "Dedicated danger models scan for guns, knives, fire, and smoke with multi-frame confirmation and per-class confidence floors.",
    points: ["Custom YOLO danger models", "Per-class confidence thresholds", "Multi-frame confirmation logic", "Instant voice and multi-channel alert"],
    label: "DangerDetector - CRITICAL",
    boxes: [["WEAPON - 0.71", "40%", "45%", "28%", "30%", "var(--danger)"]],
  },
  {
    name: "Behavior & Suspicion Analysis",
    desc: "A centroid tracker follows people and scores behaviors over time, including loitering, pacing, hesitation, and spatial anomalies.",
    points: ["Per-track suspicion scoring", "Loitering and pacing detection", "Spatial-anomaly heatmaps", "Stress-level classification"],
    label: "BehaviorAnalyzer - SuspicionScorer",
    boxes: [["ID_4 - STRESS:HIGH", "35%", "35%", "24%", "46%", "var(--warn)"]],
  },
  {
    name: "Crowd Intelligence",
    desc: "Density heatmaps and congestion grids reveal how people move through a space, flag crowd formation, and surface evacuation-speed anomalies.",
    points: ["Gaussian density heatmaps", "Congestion-zone alerts", "Crowd-formation detection", "Configurable grid resolution"],
    label: "CrowdIntelligence",
    boxes: [["CROWD - 6 PEOPLE", "20%", "20%", "60%", "60%", "var(--warn)"]],
  },
  {
    name: "Pose & Fall Detection",
    desc: "MediaPipe Pose analyzes body geometry to detect falls and raised-hands / surrender postures for safety and duress signalling.",
    points: ["Fall detection via torso vector", "Hands-raised detection", "Per-frame skeletal analysis", "Alert and voice pipeline integration"],
    label: "PoseDetector",
    boxes: [["FALL DETECTED", "30%", "25%", "50%", "24%", "var(--danger)"]],
  },
  {
    name: "Attendance Automation",
    desc: "Recognized faces are converted into attendance records after confident recognition across several frames, with automatic late calculation.",
    points: ["Hands-free clock-in / out", "Late-arrival grace windows", "12-hour auto clock-out", "CSV / report export"],
    label: "AttendanceManager",
    boxes: [["CLOCK-IN - Mikha", "30%", "28%", "26%", "44%", "var(--ok)"]],
  },
  {
    name: "AI Assistant & Alerts",
    desc: "An LLM-backed assistant answers natural-language questions over live state and event history while alert channels dispatch snapshots.",
    points: ["Database query tools", "Live camera context injection", "Email / Telegram / webhook alerts", "Priority voice announcements"],
    label: "AIAssistant - AlertManager",
    boxes: [["ASSISTANT - ONLINE", "25%", "25%", "50%", "30%", "var(--neon)"]],
  },
];

const faqs = [
  ["How does the AI actually work?", "OptiVox runs a layered pipeline per frame: YOLO detects objects and threats, InsightFace generates face embeddings, MediaPipe analyzes pose and hands, and a behavior engine scores movement over time."],
  ["Is my data stored in the cloud?", "No. By default, face embeddings, event logs, attendance, and snapshots are stored locally. Nothing leaves your machine unless you explicitly configure alerts or remote streaming."],
  ["Do I need an internet connection?", "Not for core detection. The vision pipeline can run offline. Internet is only needed for optional assistant, alert, or remote dashboard features."],
  ["Can it tell a real face from a photo or screen?", "Yes. The anti-spoofing module checks liveness before trusting recognition or attendance events."],
  ["What hardware / GPU do I need?", "OptiVox runs on CPU out of the box, but an NVIDIA GPU with CUDA is recommended for smooth multi-model real-time performance."],
  ["How reliable is the attendance system?", "A person must be confidently recognized across several consecutive frames before clock-in, suppressing false positives."],
  ["What cameras are compatible?", "Anything OpenCV can open: built-in webcams, USB cameras, and IP cameras via RTSP URLs."],
  ["What is the difference between local and web architecture?", "The local system does AI processing on your machine. The web layer streams annotated output and exposes dashboards and APIs."],
  ["How does licensing / organization access work?", "The public site is open. The private dashboard is gated behind credentials issued per organization."],
  ["Can it scale to many cameras or locations?", "The current release targets local single-camera intelligence, with multi-camera and multi-organization support on the roadmap."],
];

function useReveal() {
  const location = useLocation();
  useEffect(() => {
    const nodes = Array.from(document.querySelectorAll(".reveal"));
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry, index) => {
        if (!entry.isIntersecting) return;
        entry.target.style.transitionDelay = `${Math.min(index, 5) * 50}ms`;
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12 });
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [location.pathname]);
}

function Brand() {
  return <Link to="/" className="brand"><span className="mark"></span><b>OPTI</b><span>VOX</span></Link>;
}

function BackToTop() {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > 420);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return <button type="button" className={`back-to-top ${visible ? "visible" : ""}`} onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })} aria-label="Back to top">↑ <span>Top</span></button>;
}

function Layout({ children }) {
  const [open, setOpen] = useState(false);
  useReveal();
  const links = [["/", "Home"], ["/how-it-works", "How It Works"], ["/features", "Features"], ["/about", "About"], ["/faq", "FAQ"], ["/contact", "Contact"]];
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <div className="grid-fx"></div>
      <header className="nav">
        <div className="wrap">
          <Brand />
          <nav className={`nav-links ${open ? "open" : ""}`} id="navLinks">
            {links.map(([to, label]) => <NavLink key={to} to={to} onClick={() => setOpen(false)}>{label}</NavLink>)}
            <Link to="/login" className="btn btn-ghost" style={{ padding: ".5rem 1rem" }}>Open Dashboard</Link>
          </nav>
          <button className="nav-toggle" aria-label="Menu" onClick={() => setOpen((value) => !value)}>Menu</button>
        </div>
      </header>
      <main id="main">{children}</main>
      <BackToTop />
      <Footer />
    </>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="wrap">
        <div className="footer-grid">
          <div>
            <Brand />
            <p style={{ color: "var(--text-dim)", fontSize: ".9rem", maxWidth: 320, marginTop: "1rem" }}>
              Next-generation intelligent security infrastructure. Surveillance, analytics, and attendance powered by on-device AI.
            </p>
          </div>
          <div><h4>Product</h4><Link to="/features">Features</Link><Link to="/contact">Request Access</Link></div>
          <div><h4>Company</h4><Link to="/about">About</Link><Link to="/faq">FAQ</Link><Link to="/login">Open Dashboard</Link></div>
          <div><h4>System</h4><span>YOLOv8</span><span>InsightFace</span><span>Edge / On-Premise</span></div>
        </div>
        <div className="footer-bottom">
          <span>Copyright {year} OptiVox. Local-first AI security.</span>
          <span>STATUS: <span style={{ color: "var(--ok)" }}>OPERATIONAL</span></span>
        </div>
      </div>
    </footer>
  );
}

function DetectionVisual({ label, boxes }) {
  return (
    <div className="fvis">
      <span className="hud-label">LIVE - {label}</span>
      {boxes.map(([text, top, left, width, height, color]) => (
        <div key={text} className="detbox" data-l={text} style={{ top, left, width, height, borderColor: color }} />
      ))}
      <div className="scanline" style={{ position: "absolute" }}></div>
    </div>
  );
}

function HomePage() {
  return (
    <Layout>
      <CinematicHero />
      <section className="hero legacy-home-hero">
        <div className="wrap">
          <div className="hero-grid">
            <div className="hero-copy">
              <span className="kicker reveal">Local-first computer vision</span>
              <h1 className="reveal"><span className="chrome">Observation becomes</span><span className="line2 neon-text">understanding.</span></h1>
              <p className="lead reveal">OptiVox connects perception, recognition, attendance, security events, and human response into one explainable intelligence system for real spaces.</p>
              <div className="hero-cta reveal"><Link to="/how-it-works" className="btn btn-primary">Explore the Intelligence Map <span>→</span></Link><Link to="/features" className="btn btn-ghost">Explore Features</Link></div>
              <div className="hero-proof reveal"><span><i></i> Core processing can run locally</span><span><i></i> Built for human review</span></div>
            </div>
            <div className="hero-visual reveal">
              <div className="hero-network"><span className="hero-network-core">OPTIVOX<br /><b>CORE</b></span>{featureContent.map((feature, index) => <Link key={feature.id} to={`/features#${feature.id}`} className={`hero-feature-node hero-feature-node-${index + 1}`} style={{ "--feature-color": feature.color }}><i></i><span>{feature.title}</span></Link>)}{featureContent.map((feature, index) => <span key={`${feature.id}-line`} className={`hero-network-line feature-line-${index + 1}`} style={{ "--feature-color": feature.color }}></span>)}</div>
            </div>
          </div>
          <Stats />
        </div>
      </section>
      <section className="problem-section"><div className="wrap"><div className="section-head reveal"><span className="kicker">The gap</span><h2 className="chrome">A camera can record a moment. A system can help you understand it.</h2><p>OptiVox is built for the space between raw footage and a useful decision.</p></div><div className="problem-grid"><div className="problem-card reveal"><span className="problem-number">01</span><h3>Manual attendance</h3><p>Paper logs and repeated entry create friction, mistakes, and records that are hard to review.</p><div className="problem-flow"><span>Arrival</span><span>→</span><span className="muted">Manual entry</span><span>→</span><span className="muted">Delayed record</span></div></div><div className="problem-card reveal"><span className="problem-number">02</span><h3>Passive monitoring</h3><p>Footage without interpretation leaves people searching for meaning after the moment has passed.</p><div className="problem-flow"><span>Camera</span><span>→</span><span className="muted">Hours of footage</span><span>→</span><span className="muted">Late response</span></div></div></div></div></section>
      <section className="map-preview-section"><div className="wrap"><div className="section-head reveal"><span className="kicker">A connected system</span><h2 className="chrome">Meet the intelligence behind the interface.</h2><p>OptiVox does not treat recognition, attendance, security, and reporting as separate products. They are connected stages in one flow.</p></div><div className="preview-flow reveal"><div className="preview-flow-line"></div>{[["01", "Observe", "Camera input and visual signals"], ["02", "Recognize", "Faces, objects, and movement"], ["03", "Decide", "Confirmation and event rules"], ["04", "Remember", "Attendance and event history"], ["05", "Respond", "Dashboard, alerts, and reports"]].map(([num, title, desc]) => <div className="preview-stage" key={title}><span>{num}</span><strong>{title}</strong><small>{desc}</small></div>)}</div><div className="preview-action reveal"><Link to="/how-it-works" className="btn btn-primary">Open the full Intelligence Map <span>→</span></Link><span>Explore at your own pace or let the guided tour tell the story.</span></div></div></section>
      <Capabilities />
      <section className="privacy-band"><div className="wrap"><div className="privacy-copy reveal"><span className="kicker">Designed with boundaries</span><h2 className="chrome">Useful intelligence should remain explainable.</h2><p>Unknown does not mean dangerous. Model output is not intent. Every signal is presented with context, limitations, and a path for human review.</p><Link to="/about#privacy" className="text-link">Read the privacy and limitations notes <span>→</span></Link></div><div className="privacy-list reveal"><div><span>01</span><strong>Local by default</strong><p>Core records and embeddings can remain inside your environment.</p></div><div><span>02</span><strong>Human in the loop</strong><p>Events are prompts for review, not automatic judgments about people.</p></div><div><span>03</span><strong>Status is visible</strong><p>Core, prototype, experimental, and planned features are labelled honestly.</p></div></div></div></section>
      <section>
        <div className="wrap">
          <div className="hero-grid" style={{ alignItems: "center" }}>
            <div className="reveal">
              <span className="kicker">Conversational Intelligence</span>
              <h2 className="chrome" style={{ fontSize: "clamp(1.8rem,4vw,2.6rem)", margin: ".8rem 0 1.2rem" }}>Talk to your security system.</h2>
              <p style={{ color: "var(--text-dim)", marginBottom: "1.4rem" }}>The built-in AI assistant connects directly to live camera context and the event database.</p>
              <div className="panel" style={{ padding: "1.3rem", fontFamily: "var(--font-mono)", fontSize: ".85rem" }}>
                <div style={{ color: "var(--neon)" }}>who is currently in the building?</div>
                <div style={{ color: "var(--text-dim)", marginTop: ".5rem" }}>3 people clocked in and 1 active stranger being tracked at the Main Entrance.</div>
              </div>
            </div>
            <div className="reveal"><DetectionVisual label="CAM_0 - MAIN ENTRANCE" boxes={[["Mikha - 0.94", "30%", "22%", "24%", "42%", "var(--ok)"], ["UNKNOWN - TRACKING", "38%", "58%", "20%", "36%", "var(--warn)"]]} /></div>
          </div>
        </div>
      </section>
    </Layout>
  );
}

function Stats() {
  const values = [["17+", "AI Detection Modules"], ["local", "Core processing option"], ["5", "Connected system stages"], ["24/7", "Operational awareness"]];
  return <div className="stats">{values.map(([num, label]) => <div key={label} className="stat panel bracket reveal"><div className="num">{num}</div><div className="lbl">{label}</div></div>)}</div>;
}

function HowItWorksPage() {
  return <Layout><section className="page-intro"><div className="wrap"><span className="kicker reveal">System walkthrough</span><h1 className="chrome reveal">The OptiVox Intelligence Map</h1><p className="intro-lead reveal">Explore how visual perception becomes recognition, memory, and purposeful action.</p><div className="intro-meta reveal"><span>18 connected nodes</span><span>4 flow presets</span><span>1 guided story</span></div></div></section><div className="wrap"><IntelligenceMap /></div><section className="map-notes"><div className="wrap"><div className="section-head reveal"><span className="kicker">How to read it</span><h2 className="chrome">The map has three layers of detail.</h2></div><div className="notes-grid"><div className="note-card reveal"><span>01</span><h3>Label</h3><p>See the public name of each module and its place in the flow.</p></div><div className="note-card reveal"><span>02</span><h3>Purpose</h3><p>Select a node to understand its input, process, output, technology, and limitation.</p></div><div className="note-card reveal"><span>03</span><h3>Story</h3><p>Follow a preset or guided tour to watch one event move through the system.</p></div></div></div></section></Layout>;
}

function Capabilities() {
  return (
    <section>
      <div className="wrap">
        <div className="section-head reveal"><span className="kicker">Core Intelligence</span><h2 className="chrome">One camera. Seventeen kinds of awareness.</h2><p>Every frame is analyzed by specialized models running together in real time.</p></div>
        <div className="cards">{capabilities.map(([name, desc, tag]) => <div className="card reveal" key={name}><div className="ic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="3" /><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" /></svg></div><span className="tag">{tag}</span><h3>{name}</h3><p>{desc}</p></div>)}</div>
      </div>
    </section>
  );
}

function FeaturesPage() {
  return <Layout><section className="page-intro features-intro"><div className="wrap"><span className="kicker reveal">Capability guide</span><h1 className="chrome reveal">Eight ways OptiVox turns vision into useful context.</h1><p className="intro-lead reveal">Start with the human explanation, then open the mechanism, status, and limitation behind each capability.</p></div></section><section className="features-navigator-section"><div className="wrap"><FeatureNavigator /></div></section><section className="feature-details"><div className="wrap">{featureContent.map((feature, index) => <article className="feature-detail reveal" id={feature.id} key={feature.id}><div className="feature-detail-number" style={{ color: feature.color }}>0{index + 1}</div><div className="feature-detail-copy"><span className="tag" style={{ color: feature.color, borderColor: `${feature.color}55` }}>{feature.tag}</span><h2 className="chrome">{feature.title}</h2><p className="feature-plain">{feature.plain}</p><div className="feature-detail-grid"><div><h4>Why it matters</h4><p>{feature.why}</p></div><div><h4>How it works</h4><p>{feature.how}</p></div><div><h4>Input <span>→</span> Process <span>→</span> Output</h4><p className="feature-mono">{feature.input}<br />{feature.process}<br />{feature.output}</p></div><div><h4>Technology and status</h4><p className="feature-mono">{feature.tech}<br /><span style={{ color: feature.color }}>{feature.status}</span></p></div></div><div className="feature-limitation"><strong>Limitation</strong><span>{feature.limitation}</span></div><p className="feature-related">Related capability: <a href={`#${featureContent.find((item) => item.title === feature.related)?.id || feature.id}`}>{feature.related}</a></p></div><div className="feature-evidence"><DetectionVisual label={feature.tag} boxes={[[feature.status.toUpperCase(), "30%", "24%", "48%", "38%", feature.color]]} /><span>{feature.status}</span></div></article>)}</div></section><section className="feature-final"><div className="wrap"><span className="kicker">See the system in motion</span><h2 className="chrome">A capability is only useful when you can follow its path.</h2><Link to="/how-it-works" className="btn btn-primary">Open the Intelligence Map <span>→</span></Link></div></section></Layout>;
}

function DemoPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [started, setStarted] = useState(false);
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState({ fps: "0.0", faces: 0, objects: 0, live: "-", threat: "NOMINAL" });
  const [feed, setFeed] = useState(["awaiting sandbox start"]);

  useEffect(() => {
    if (!started) return undefined;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let running = true;
    let lastT = performance.now();
    let fps = 0;
    let frameN = 0;
    let boxes = [];
    let prevFrame = null;

    function detectMotion() {
      const w = 64;
      const h = 36;
      const tmp = document.createElement("canvas");
      tmp.width = w;
      tmp.height = h;
      const tctx = tmp.getContext("2d");
      tctx.drawImage(video, 0, 0, w, h);
      const cur = tctx.getImageData(0, 0, w, h).data;
      const out = [];
      if (prevFrame) {
        let minX = w, minY = h, maxX = 0, maxY = 0, count = 0;
        for (let y = 0; y < h; y += 1) {
          for (let x = 0; x < w; x += 1) {
            const i = (y * w + x) * 4;
            const delta = Math.abs(cur[i] - prevFrame[i]) + Math.abs(cur[i + 1] - prevFrame[i + 1]) + Math.abs(cur[i + 2] - prevFrame[i + 2]);
            if (delta > 70) { count += 1; minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
          }
        }
        if (count > 36 && maxX > minX && maxY > minY) out.push({ x: (minX / w) * canvas.width, y: (minY / h) * canvas.height, w: ((maxX - minX) / w) * canvas.width, h: ((maxY - minY) / h) * canvas.height, conf: Math.min(0.98, 0.6 + count / 700) });
      }
      prevFrame = cur;
      return out;
    }

    function drawBox(box, label) {
      ctx.strokeStyle = "#34e0a1";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#34e0a1";
      ctx.shadowBlur = 8;
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.shadowBlur = 0;
      ctx.save();
      ctx.scale(-1, 1);
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(-(box.x + box.w), box.y - 18, Math.max(120, box.w), 16);
      ctx.fillStyle = "#34e0a1";
      ctx.font = "11px Space Mono, monospace";
      ctx.fillText(label, -(box.x + box.w) + 4, box.y - 6);
      ctx.restore();
    }

    function loop() {
      if (!running) return;
      const now = performance.now();
      fps = fps * 0.9 + (1000 / Math.max(now - lastT, 1)) * 0.1;
      lastT = now;
      frameN += 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (frameN % 4 === 0) boxes = detectMotion();
      boxes.forEach((box, index) => drawBox(box, `TRACK_${index + 1} ${box.conf.toFixed(2)}`));
      setMetrics({ fps: fps.toFixed(1), faces: boxes.length ? 1 : 0, objects: boxes.length, live: boxes.length ? "CHECKING" : "-", threat: "NOMINAL" });
      if (frameN % 100 === 0 && boxes.length) setFeed((items) => [`${new Date().toLocaleTimeString("en-GB")} - LEGACY FALLBACK`, ...items].slice(0, 12));
      requestAnimationFrame(loop);
    }

    loop();
    return () => { running = false; };
  }, [started]);

  async function startDemo() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      canvasRef.current.width = video.videoWidth || 960;
      canvasRef.current.height = video.videoHeight || 540;
      setFeed([`${new Date().toLocaleTimeString("en-GB")} - SANDBOX INITIALIZED`]);
      setStarted(true);
    } catch (err) {
      setError(`Camera unavailable: ${err.message || err.name}`);
    }
  }

  return (
    <Layout>
      <section style={{ paddingTop: "4rem" }}>
        <div className="wrap">
          <div className="section-head reveal"><span className="kicker">Sandbox Environment</span><h2 className="chrome">See the HUD in action.</h2><p>This browser-side sandbox uses your webcam locally to render the OptiVox overlay. Nothing is uploaded or stored.</p></div>
          <div className="demo-stage">
            <div className="video-shell">
              <video id="demoVideo" ref={videoRef} autoPlay muted playsInline></video><canvas id="demoCanvas" ref={canvasRef}></canvas>
              <div className="video-overlay"><div className="hud-corner tl"></div><div className="hud-corner tr"></div><div className="hud-corner bl"></div><div className="hud-corner br"></div><div className="hud-top"><span>{new Date().toLocaleTimeString("en-GB")}</span><span className="rec"><span className="dot"></span> SANDBOX</span></div></div>
              {!started && <div className="demo-prompt"><div><h3 className="chrome" style={{ fontSize: "1.4rem", marginBottom: ".8rem" }}>Camera Access Required</h3><p style={{ color: "var(--text-dim)", marginBottom: "1.4rem", maxWidth: 320 }}>Grant camera permission to preview the detection overlay on your feed.</p><button className="btn btn-primary" onClick={startDemo}>Start Sandbox</button><p style={{ color: "var(--danger)", fontSize: ".82rem", marginTop: "1rem" }}>{error}</p></div></div>}
            </div>
            <div className="demo-side">
              <MetricPanel metrics={metrics} />
              <div className="panel"><h4 style={panelTitleStyle}>Detection Feed</h4><div className="det-feed">{feed.map((item) => <div className="line" key={item}>{item}</div>)}</div></div>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}

const panelTitleStyle = { fontFamily: "var(--font-mono)", fontSize: ".72rem", letterSpacing: ".16em", color: "var(--text-faint)", textTransform: "uppercase", marginBottom: ".8rem" };

function MetricPanel({ metrics }) {
  return <div className="panel"><h4 style={panelTitleStyle}>System Telemetry</h4>{[["Pipeline FPS", metrics.fps], ["Faces in frame", metrics.faces], ["Objects tracked", metrics.objects], ["Liveness", metrics.live], ["Threat level", metrics.threat]].map(([label, value]) => <div className="metric-row" key={label}><span>{label}</span><span className="v">{value}</span></div>)}</div>;
}

function AboutPage() {
  const stack = [["YOLOv8", "Object, weapon, and fire detection"], ["InsightFace", "512-d face embeddings and recognition"], ["MediaPipe", "Pose, hand, and gesture analysis"], ["FAISS", "High-speed embedding search"], ["OpenCV", "Frame capture and rendering"], ["SQLite", "Local event and attendance store"]];
  return (
    <Layout>
      <section style={{ paddingTop: "5rem" }}><div className="wrap">
        <div className="section-head reveal"><span className="kicker">Why OptiVox Exists</span><h2 className="chrome">Surveillance should understand, not just record.</h2><p>OptiVox was built to turn passive footage into timely awareness.</p></div>
        <div className="cards" style={{ gridTemplateColumns: "1fr 1fr" }}><div className="card reveal"><span className="tag">MISSION</span><h3>Make every camera intelligent</h3><p>Recognize people, detect threats, track attendance, and reason about behavior in real time.</p></div><div className="card reveal"><span className="tag">VISION</span><h3>Security that respects privacy</h3><p>Run locally so faces, footage, and analytics stay inside your walls by default.</p></div></div>
        <div style={{ marginTop: "4rem" }} className="reveal"><span className="kicker">Built On</span><h2 className="chrome" style={{ fontSize: "clamp(1.6rem,3.5vw,2.3rem)", margin: ".8rem 0 1.5rem" }}>A modern computer-vision stack</h2><div className="panel" style={{ padding: "2rem" }}><div className="form-grid" style={{ gap: "1.5rem" }}>{stack.map(([name, desc]) => <div key={name}><div className="neon-text" style={{ fontFamily: "var(--font-display)", fontWeight: 600, fontSize: "1.1rem" }}>{name}</div><div style={{ color: "var(--text-dim)", fontSize: ".88rem" }}>{desc}</div></div>)}</div></div></div>
        <div id="privacy" className="panel bracket reveal" style={{ marginTop: "4rem", padding: "2rem", background: "rgba(255,181,71,.05)", borderColor: "rgba(255,181,71,.25)" }}><span className="kicker" style={{ color: "var(--warn)" }}>Ethics & Responsible Use</span><h3 className="chrome" style={{ margin: ".8rem 0" }}>Powerful tools demand clear boundaries</h3><p style={{ color: "var(--text-dim)" }}>Facial recognition and behavioral analysis carry privacy implications. Unknown does not mean dangerous, and observable movement is not criminal intent. Operators are responsible for lawful, transparent, consent-aware, and proportionate deployment.</p></div>
      </div></section>
    </Layout>
  );
}

function FaqPage() {
  const [open, setOpen] = useState(null);
  return <Layout><section style={{ paddingTop: "5rem" }}><div className="wrap" style={{ maxWidth: 860 }}><div className="section-head reveal"><span className="kicker">Frequently Asked</span><h2 className="chrome">Questions, answered.</h2><p>How OptiVox works, what it stores, and what it needs to run.</p></div>{faqs.map(([q, a], index) => <div className={`faq-item reveal ${open === index ? "open" : ""}`} key={q}><button className="faq-q" onClick={() => setOpen(open === index ? null : index)}>{q}<span className="pm">+</span></button><div className="faq-a"><div className="faq-a-inner">{a}</div></div></div>)}<div className="panel bracket reveal" style={{ marginTop: "2.5rem", padding: "2rem", textAlign: "center" }}><h3 className="chrome" style={{ marginBottom: ".6rem" }}>Still have questions?</h3><p style={{ color: "var(--text-dim)", marginBottom: "1.4rem" }}>Reach the team for technical details, demos, or deployment guidance.</p><Link to="/contact" className="btn btn-primary">Contact Us</Link></div></div></section></Layout>;
}

function ContactPage() {
  const [status, setStatus] = useState("");
  async function submit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = Object.fromEntries(new FormData(form));
    setStatus("transmitting request...");
    try {
      const out = await apiRequest("/api/v1/contact", { method: "POST", body: JSON.stringify(data) });
      setStatus(out?.message || "Request received. We will be in touch.");
      form.reset();
    } catch (err) {
      setStatus(err.message || "Could not submit. Please email us directly.");
    }
  }
  return <Layout><section style={{ paddingTop: "5rem" }}><div className="wrap" style={{ maxWidth: 920 }}><div className="section-head reveal"><span className="kicker">Get Started</span><h2 className="chrome">Request access or a demo.</h2><p>Tell us about your deployment. We will follow up with access, installation help, or API onboarding.</p></div><div className="panel reveal" style={{ padding: "2.2rem" }}><form onSubmit={submit}><div className="form-grid"><div className="field"><label>Full name</label><input type="text" name="name" required placeholder="Jane Operator" /></div><div className="field"><label>Work email</label><input type="email" name="email" required placeholder="jane@org.com" /></div><div className="field"><label>Organization</label><input type="text" name="org" placeholder="Acme Security Co." /></div><div className="field"><label>Request type</label><select name="type"><option>Enterprise access</option><option>Live demo</option><option>Installation support</option><option>API access</option></select></div><div className="field full"><label>Deployment details</label><textarea name="message" placeholder="How many cameras? What environment? Any GPU available?"></textarea></div></div><button type="submit" className="btn btn-primary" style={{ marginTop: ".5rem" }}>Submit Request</button><p style={{ marginTop: "1rem", fontFamily: "var(--font-mono)", fontSize: ".85rem", color: "var(--ok)" }}>{status}</p></form></div><div className="cards" style={{ gridTemplateColumns: "repeat(3,1fr)", marginTop: "2rem" }}><div className="card reveal"><span className="tag">DEMO</span><h3>See it live</h3><p>Walk through the detection suite on real footage.</p></div><div className="card reveal"><span className="tag">INSTALL</span><h3>On-site setup</h3><p>Guidance on cameras, GPU, models, and tuning.</p></div><div className="card reveal"><span className="tag">API</span><h3>Integrate</h3><p>Connect events and attendance to your systems.</p></div></div></div></section></Layout>;
}

function LoginPage() {
  const [error, setError] = useState("");
  const navigate = useNavigate();
  async function submit(event) {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.currentTarget));
    setError("");
    try {
      await apiRequest("/api/v1/auth/login", { method: "POST", body: JSON.stringify(data) });
      navigate("/dashboard");
    } catch (err) {
      setError(err.message || "Invalid credentials");
    }
  }
  return <Layout><section className="page-center" style={{ paddingTop: "6rem" }}><div className="wrap" style={{ maxWidth: 420 }}><div className="panel bracket reveal" style={{ padding: "2.5rem" }}><div style={{ textAlign: "center", marginBottom: "1.8rem" }}><span className="kicker" style={{ justifyContent: "center" }}>Restricted</span><h2 className="chrome" style={{ fontSize: "1.6rem", marginTop: ".6rem" }}>Command Center Access</h2><p style={{ color: "var(--text-dim)", fontSize: ".9rem", marginTop: ".5rem" }}>Authorized organizations only.</p></div><form onSubmit={submit}><div className="field"><label>Username</label><input type="text" name="username" required autoFocus /></div><div className="field"><label>Password</label><input type="password" name="password" required /></div><p className="login-status">{error}</p><button type="submit" className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }}>Authenticate</button></form><p style={{ color: "var(--text-faint)", fontSize: ".78rem", marginTop: "1.4rem", textAlign: "center", fontFamily: "var(--font-mono)" }}>Demo credentials: <span className="neon-text">admin / optivox</span></p></div></div></section></Layout>;
}

function DashboardPage() {
  const [ready, setReady] = useState(false);
  const [bridgeMode, setBridgeMode] = useState("CONNECTING");
  const [stats, setStats] = useState({});
  const [state, setState] = useState({});
  const [events, setEvents] = useState([]);
  const [attendance, setAttendance] = useState([]);
  const [people, setPeople] = useState([]);
  const [msg, setMsg] = useState("");
  const [assistantAnswer, setAssistantAnswer] = useState("Assistant answers appear here when your vision module is configured.");
  const navigate = useNavigate();

  useEffect(() => {
    apiRequest("/api/v1/auth/me").then(() => setReady(true)).catch(() => navigate("/login"));
  }, [navigate]);

  useEffect(() => {
    if (!ready) return undefined;
    let alive = true;
    async function refreshAll() {
      try {
        const [statsOut, stateOut, eventsOut, attendanceOut, peopleOut] = await Promise.all([
          apiRequest("/api/v1/stats"), apiRequest("/api/v1/live-state"), apiRequest("/api/v1/events?limit=30"), apiRequest("/api/v1/attendance"), apiRequest("/api/v1/people"),
        ]);
        if (!alive) return;
        setStats(statsOut || {});
        setState(stateOut || {});
        setEvents(eventsOut?.events || []);
        setAttendance(attendanceOut?.records || []);
        setPeople(peopleOut?.people || []);
        setBridgeMode((stateOut?.bridge?.mode || "unknown").toUpperCase());
      } catch (_) {
        if (alive) setBridgeMode("OFFLINE");
      }
    }
    refreshAll();
    const fast = setInterval(refreshAll, 3500);
    return () => { alive = false; clearInterval(fast); };
  }, [ready]);

  async function logout() {
    await apiRequest("/api/v1/auth/logout", { method: "POST", body: "{}" }).catch(() => null);
    navigate("/");
  }

  async function postAction(endpoint, data, success) {
    try {
      const out = await apiRequest(endpoint, { method: "POST", body: JSON.stringify(data || {}) });
      setMsg(success(out));
    } catch (err) {
      setMsg(err.message || "Action failed");
    }
  }

  if (!ready) return <div className="dash-body"><div className="grid-fx"></div><div className="not-found"><span className="kicker">Checking Access</span></div></div>;

  const frame = state.frame || {};
  const toggles = state.toggles || {};
  return (
    <div className="dash-body">
      <div className="grid-fx"></div>
      <div className="dash">
        <aside className="dash-side"><Brand /><nav className="dash-nav"><a href="#live" className="active">Live Feed</a><a href="#detections">Detections</a><a href="#controls">Controls</a><a href="#attendance">Attendance</a><a href="#assistant">Assistant</a><button className="nav-link-button" onClick={logout}>Sign out</button></nav></aside>
        <main className="dash-main">
          <div className="dash-topbar"><div><span className="kicker">Command Center</span><h2 className="chrome">Live Monitoring</h2></div><span className="live-badge"><span className="dot"></span><span>{bridgeMode}</span></span></div>
          <div className="kpi-row"><Kpi value={stats.enrolled_faces} label="Enrolled Faces" /><Kpi value={stats.events_today} label="Events Today" /><Kpi value={stats.clocked_in} label="Clocked In" /><Kpi value={stats.active_strangers} label="Active Strangers" /><Kpi value={frame.process_ms} label="Process ms" /></div>
          <section id="live" className="dash-grid">
            <div className="stream-shell"><img src={apiUrl("/api/v1/video-feed")} alt="Live OptiVox feed" /><div className="video-overlay"><div className="hud-corner tl"></div><div className="hud-corner tr"></div><div className="hud-corner bl"></div><div className="hud-corner br"></div></div></div>
            <div id="detections" className="panel dash-panel"><h4>Live Detections</h4><div className="split-lists"><Feed title="Faces" items={frame.faces} empty="no faces visible" format={(f) => `${f.name || f.label || "UNKNOWN"} - ${Number(f.confidence || 0).toFixed(2)}${f.is_real === false ? " - SPOOF" : ""}`} /><Feed title="Objects" items={frame.objects} empty="no objects detected" format={(o) => `${o.class_name || "object"} - ${Number(o.confidence || 0).toFixed(2)} - ${o.category || "general"}`} /></div><Feed title="Held Objects" items={frame.held_objects} empty="no held objects confirmed" format={(o) => `${o.class_name || "object"} - ${Number(o.confidence || 0).toFixed(2)}`} short /></div>
          </section>
          <section className="dash-grid lower"><div id="controls" className="panel dash-panel"><h4>Operator Controls</h4><ToggleGrid toggles={toggles} onToggle={(name, value) => postAction("/api/v1/toggle", { name, value }, (out) => out.message || "updated")} /><NameAction onSubmit={(name) => postAction("/api/v1/enroll-visible-face", { name }, (out) => out.message || "enrolled")} /><p className="action-msg">{msg}</p></div><div className="panel dash-panel"><h4>Event Timeline</h4><List items={events} empty="no events recorded yet" format={(e) => `${timeAgo(e.timestamp)} - ${e.event_type}${e.person ? ` - ${e.person}` : ""}${e.location ? ` - ${e.location}` : ""}`} /></div></section>
          <section id="attendance" className="dash-grid lower"><div className="panel dash-panel"><h4>Today's Attendance</h4><List items={attendance} empty="no attendance records today" format={(r) => `${r.name} - in ${(r.clock_in || "-").slice(11, 16) || "-"} - out ${(r.clock_out || "-").slice(11, 16) || "-"}${r.late_minutes ? ` - ${r.late_minutes}m late` : ""}`} /></div><div className="panel dash-panel"><h4>People & Manual Attendance</h4><PeopleControls people={people} onClock={(kind, name) => postAction(kind === "in" ? "/api/v1/manual-clock-in" : "/api/v1/manual-clock-out", { name }, (out) => `${name}: ${JSON.stringify(out.result)}`)} /><List items={people} empty="no enrolled people" format={(p) => `${p.name}${p.role ? ` - ${p.role}` : ""}`} /></div></section>
          <section id="assistant" className="panel dash-panel assistant-panel"><h4>AI Assistant</h4><AssistantAsk onAsk={(question) => postAction("/api/v1/assistant/ask", { question }, (out) => { setAssistantAnswer(out.answer || "No answer."); return "assistant answered"; })} /><div className="assistant-answer">{assistantAnswer}</div></section>
        </main>
      </div>
      <BackToTop />
    </div>
  );
}

function Kpi({ value, label }) {
  return <div className="kpi panel bracket"><div className="v">{value ?? "-"}</div><div className="l">{label}</div></div>;
}

function Feed({ title, items, empty, format, short }) {
  return <div><h5>{title}</h5><div className={`det-feed ${short ? "short" : ""}`}><List items={items} empty={empty} format={format} /></div></div>;
}

function List({ items, empty, format }) {
  if (!items || !items.length) return <div className="line">{empty}</div>;
  return <>{items.map((item, index) => <div className="line" key={item.id || item.name || index}>{format(item)}</div>)}</>;
}

const toggleLabels = {
  show_object_boxes: "Object boxes",
  show_heatmap: "Heatmap",
  show_pose_landmarks: "Pose overlay",
  show_age_gender: "Age/gender",
  show_zones_grid: "Zones grid",
  danger_detection: "Danger detection",
};

function ToggleGrid({ toggles, onToggle }) {
  return <div className="toggle-grid">{Object.entries(toggleLabels).map(([name, label]) => <label className="toggle" key={name}><span>{label}</span><input type="checkbox" checked={!!toggles[name]} onChange={(event) => onToggle(name, event.target.checked)} /></label>)}</div>;
}

function NameAction({ onSubmit }) {
  const [name, setName] = useState("");
  return <div className="action-row"><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Name visible unknown" /><button className="btn btn-primary compact" onClick={() => { if (name.trim()) { onSubmit(name.trim()); setName(""); } }}>Enroll Visible Face</button></div>;
}

function PeopleControls({ people, onClock }) {
  const [selected, setSelected] = useState("");
  useEffect(() => { if (!selected && people?.length) setSelected(people[0].name); }, [people, selected]);
  return <div className="action-row"><select value={selected} onChange={(event) => setSelected(event.target.value)}>{(people || []).map((person) => <option value={person.name} key={person.name}>{person.name}</option>)}</select><button className="btn btn-ghost compact" onClick={() => selected && onClock("in", selected)}>Clock In</button><button className="btn btn-ghost compact" onClick={() => selected && onClock("out", selected)}>Clock Out</button></div>;
}

function AssistantAsk({ onAsk }) {
  const [question, setQuestion] = useState("");
  return <div className="assistant-row"><input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && question.trim()) onAsk(question.trim()); }} placeholder="Ask: who is visible, what objects do you see, who is clocked in?" /><button className="btn btn-primary compact" onClick={() => question.trim() && onAsk(question.trim())}>Ask</button></div>;
}

function ExpandedAboutPage() {
  const stack = [["InsightFace", "Face detection, landmarks, and recognition embeddings"], ["FAISS", "Fast similarity search across enrolled identities"], ["YOLO", "Selected object and safety-event detection"], ["MediaPipe", "Pose and hand landmark analysis"], ["OpenCV", "Camera capture, frame processing, and overlays"], ["SQLite", "Local people, attendance, events, and alerts"], ["Flask API", "Structured bridge between the local engine and web interface"], ["React", "Dashboard and public explanation layer"]];
  const milestones = ["Problem discovery", "Research and architecture", "Database foundation", "Face recognition", "Attendance automation", "Security modules", "Web dashboard", "Exhibition integration"];
  return <Layout><section className="page-intro about-intro"><div className="wrap"><span className="kicker reveal">Project story</span><h1 className="chrome reveal">Surveillance should understand, not just record.</h1><p className="intro-lead reveal">OptiVox explores how an ordinary camera can become a local operational system that recognises enrolled people, automates attendance, and surfaces observable events for human review.</p></div></section><section className="about-story"><div className="wrap"><div className="about-lead reveal"><span className="kicker">Why OptiVox exists</span><h2 className="chrome">Observation is incomplete when it produces only footage.</h2><p>Attendance and security systems are often separated, even though both begin with the same question: who is present, what is happening, and what response is required? OptiVox was developed around the idea that perception becomes useful when it can be interpreted, remembered, and transformed into responsible action.</p></div><div className="about-principles"><div className="about-principle reveal"><span>01</span><h3>Local first</h3><p>Core processing and records can stay inside the environment that produces them.</p></div><div className="about-principle reveal"><span>02</span><h3>Human review</h3><p>Observable signals support people; they do not decide intent or danger.</p></div><div className="about-principle reveal"><span>03</span><h3>Transparent limits</h3><p>Core, prototype, experimental, and planned features are labelled honestly.</p></div></div></div></section><section className="about-architecture"><div className="wrap"><div className="section-head reveal"><span className="kicker">System architecture</span><h2 className="chrome">One flow, from camera to response.</h2></div><div className="architecture-flow reveal">{[["01", "Camera", "Live scene"], ["02", "Vision engine", "Detection and embeddings"], ["03", "Decision logic", "Thresholds and event rules"], ["04", "Local memory", "SQLite records"], ["05", "Web interface", "Dashboard and reports"]].map(([number, title, description]) => <div className="architecture-stage" key={title}><span>{number}</span><strong>{title}</strong><small>{description}</small></div>)}</div></div></section><section className="about-development"><div className="wrap"><div className="section-head reveal"><span className="kicker">Development journey</span><h2 className="chrome">Built in layers, tested through constraints.</h2></div><div className="milestone-grid">{milestones.map((milestone, index) => <div className="milestone reveal" key={milestone}><span>0{index + 1}</span><strong>{milestone}</strong></div>)}</div></div></section><section className="about-stack"><div className="wrap"><div className="section-head reveal"><span className="kicker">Technology choices</span><h2 className="chrome">Open tools, connected carefully.</h2></div><div className="stack-grid">{stack.map(([name, description]) => <div className="stack-item reveal" key={name}><strong>{name}</strong><p>{description}</p></div>)}</div></div></section><section className="about-impact"><div className="wrap"><div className="about-impact-copy reveal"><span className="kicker">SDG 9 / Industry, innovation, and infrastructure</span><h2 className="chrome">Intelligent infrastructure should be more accessible.</h2><p>OptiVox aligns with SDG 9 by demonstrating that useful AI infrastructure can be assembled from open-source software, a standard webcam, and a personal computer. Its value is not only automation, but making experimentation with intelligent infrastructure possible in environments that cannot depend on expensive proprietary systems.</p></div><div className="about-impact-facts reveal"><span>Open-source building blocks</span><span>Standard camera input</span><span>Local processing option</span><span>Designed for learning</span></div></div></section><section className="about-privacy" id="privacy"><div className="wrap"><div className="section-head reveal"><span className="kicker">Privacy and limitations</span><h2 className="chrome">Useful systems need boundaries.</h2></div><div className="limitation-grid"><div className="limitation-item reveal"><strong>Consent and access</strong><p>Enrollment should be consent-based, and biometric representations need access control and retention rules.</p></div><div className="limitation-item reveal"><strong>Recognition conditions</strong><p>Lighting, angle, occlusion, camera placement, and enrollment quality affect performance.</p></div><div className="limitation-item reveal"><strong>Experimental signals</strong><p>Liveness and pose-based events can produce false positives and remain subject to human review.</p></div><div className="limitation-item reveal"><strong>Evaluation scope</strong><p>The current project is a small-scale prototype that needs broader testing before unrestricted deployment.</p></div></div></div></section><section className="about-roadmap"><div className="wrap"><span className="kicker">Future direction</span><h2 className="chrome">The next questions are practical.</h2><div className="roadmap-list"><span>Stronger liveness safeguards</span><span>Multi-camera operation</span><span>Role-based access</span><span>Larger-scale evaluation</span><span>Formal privacy policies</span><span>Better report generation</span></div></div></section></Layout>;
}

const publicFaqs = [
  ["Product", "What does OptiVox do?", "OptiVox connects local computer vision with attendance records, observable event handling, snapshots, alerts, and a web dashboard."],
  ["Product", "Is it an attendance system or a security system?", "Both workflows begin with the same camera input. Enrolled identities can become attendance, while unresolved identities and selected observable events can become reviewable security context."],
  ["Product", "What happens when someone is unknown?", "They receive a temporary tracking identity rather than being counted as attendance. Unknown does not mean dangerous."],
  ["Technology", "Does it require internet?", "Core local detection and storage can run without internet. Optional alerts, remote access, and assistant integrations may need connectivity."],
  ["Technology", "What hardware is required?", "The prototype can run with a standard webcam and computer. A stronger CPU or compatible GPU can improve real-time performance."],
  ["Technology", "Why does recognition require several frames?", "One frame can be blurred, obstructed, or misaligned. Repeated confirmation makes the decision less dependent on a single moment."],
  ["Privacy", "Where is biometric data stored?", "The local prototype stores identity profiles and embeddings in its local data layer. Each deployment still needs its own consent, access, retention, and backup policy."],
  ["Privacy", "Is camera footage uploaded?", "The local application can keep core processing and records on the local machine unless remote channels are configured."],
  ["Reliability", "Can a photograph fool the system?", "The prototype includes experimental liveness safeguards, but they are not definitive protection and should not be presented as infallible."],
  ["Reliability", "Is it ready for full school deployment?", "It is a working project prototype. Broader evaluation, formal privacy policies, access controls, and operational testing are still required."],
  ["Product", "Where can I see the operational dashboard?", "Use Open Dashboard in the navigation to access the configured application. It may use sample or live data depending on the deployment."],
];

function AccessibleFaqPage() {
  const [open, setOpen] = useState(null); const [category, setCategory] = useState("All"); const categories = ["All", ...new Set(publicFaqs.map(([group]) => group))]; const filtered = publicFaqs.filter(([group]) => category === "All" || category === group);
  return <Layout><section className="page-intro faq-intro"><div className="wrap"><span className="kicker reveal">Questions, answered</span><h1 className="chrome reveal">A clear answer is part of the product.</h1><p className="intro-lead reveal">Learn what OptiVox does, how the prototype works, where data stays, and what still needs further validation.</p></div></section><section className="faq-section"><div className="wrap"><div className="faq-categories" role="group" aria-label="FAQ categories">{categories.map((item) => <button type="button" className={category === item ? "active" : ""} key={item} onClick={() => { setCategory(item); setOpen(null); }}>{item}</button>)}</div><div className="faq-list">{filtered.map(([group, question, answer], index) => { const id = `${category}-${index}`; const expanded = open === id; return <div className={`faq-item faq-item-repaired ${expanded ? "open" : ""}`} key={question}><button type="button" className="faq-q" aria-expanded={expanded} aria-controls={`answer-${id}`} onClick={() => setOpen(expanded ? null : id)}><span><small>{group}</small>{question}</span><span className="pm" aria-hidden="true">{expanded ? "−" : "+"}</span></button><div className="faq-a" id={`answer-${id}`} role="region"><div className="faq-a-inner">{answer}</div></div></div>; })}</div><div className="faq-action"><h2 className="chrome">Still have a project question?</h2><Link to="/contact" className="btn btn-primary">Contact the OptiVox team</Link></div></div></section></Layout>;
}

function timeAgo(iso) {
  if (!iso) return "-";
  const clean = String(iso).endsWith("Z") ? iso : `${iso}Z`;
  const seconds = (Date.now() - new Date(clean).getTime()) / 1000;
  if (!Number.isFinite(seconds)) return iso;
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function NotFound() {
  return <Layout><div className="not-found"><div><span className="kicker">404</span><h1 className="chrome">Page not found</h1><Link to="/" className="btn btn-primary" style={{ marginTop: "1rem" }}>Return Home</Link></div></div></Layout>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/how-it-works" element={<HowItWorksPage />} />
      <Route path="/features" element={<FeaturesPage />} />
      <Route path="/about" element={<ExpandedAboutPage />} />
      <Route path="/faq" element={<AccessibleFaqPage />} />
      <Route path="/contact" element={<ContactPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
