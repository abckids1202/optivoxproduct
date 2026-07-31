import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

const stages = [
  ["OBSERVATION", "A local camera receives the scene."],
  ["PERCEPTION", "Faces, people, poses, and objects become signals."],
  ["RECOGNITION", "Visual evidence is compared with enrolled profiles."],
  ["DECISION", "Repeated frames and event rules are evaluated."],
  ["MEMORY", "Relevant attendance and event context is preserved."],
  ["RESPONSE", "The dashboard, reports, and alerts become useful."],
];

const featureLinks = [
  ["face-recognition", "Recognition"],
  ["automated-attendance", "Attendance"],
  ["unknown-monitoring", "Security review"],
  ["local-data", "Local data"],
  ["alerts-evidence", "Alerts"],
  ["dashboard-assistant", "Dashboard"],
];

export default function CinematicHero() {
  const heroRef = useRef(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const stage = stages[stageIndex];

  useEffect(() => {
    if (paused || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;
    const timer = window.setInterval(() => setStageIndex((value) => (value + 1) % stages.length), 1800);
    return () => window.clearInterval(timer);
  }, [paused]);

  function handlePointerMove(event) {
    const rect = heroRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
    heroRef.current.style.setProperty("--hero-pointer-x", `${x.toFixed(3)}`);
    heroRef.current.style.setProperty("--hero-pointer-y", `${y.toFixed(3)}`);
  }

  function resetPointer() {
    heroRef.current?.style.setProperty("--hero-pointer-x", "0");
    heroRef.current?.style.setProperty("--hero-pointer-y", "0");
  }

  return <section className={`cinematic-hero ${stageIndex === stages.length - 1 ? "is-final" : ""}`} ref={heroRef} onPointerMove={handlePointerMove} onPointerLeave={resetPointer} aria-labelledby="cinematic-hero-title">
    <img className="cinematic-hero-image" src="/images/hero/optivox-hero-poster.png" alt="" aria-hidden="true" />
    <div className="cinematic-hero-shade"></div><div className="cinematic-hero-grid"></div><div className="cinematic-hero-scan"></div>
    <div className="cinematic-hero-top"><span className="cinematic-kicker"><i></i> Local-first computer vision</span><button type="button" className="cinematic-motion-toggle" onClick={() => setPaused((value) => !value)} aria-pressed={paused}>{paused ? "Resume motion" : "Pause motion"}</button></div>
    <div className="cinematic-hero-center"><div className="cinematic-stage-label"><span className="stage-pulse"></span><span>{stage[0]}</span><small>{stage[1]}</small></div><p className="cinematic-eyebrow">AI Attendance and Security System</p><h1 id="cinematic-hero-title">OPTI<span>VOX</span></h1><p className="cinematic-subtitle">A camera becomes a system that observes, recognises, remembers, and responds.</p><p className="cinematic-philosophy">Observation becomes understanding.<br /><span>Understanding leads to action.</span></p><div className="cinematic-actions"><Link to="/how-it-works" className="btn btn-primary">Enter the System <span>→</span></Link><Link to="/demo" className="btn btn-ghost">Try the Browser Demo</Link></div></div>
    <div className="cinematic-stage-track" aria-label="OptiVox system stages">{stages.map(([label], index) => <button type="button" key={label} className={index === stageIndex ? "active" : index < stageIndex ? "complete" : ""} onClick={() => setStageIndex(index)}><i></i><span>{label}</span></button>)}</div>
    <div className="cinematic-feature-links">{featureLinks.map(([id, label]) => <Link key={id} to={`/features#${id}`}>{label}<span>↗</span></Link>)}</div><div className="cinematic-scroll-cue"><span></span> Scroll to follow the system</div>
  </section>;
}
