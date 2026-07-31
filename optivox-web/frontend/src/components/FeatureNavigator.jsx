import { useEffect, useState } from "react";
import { featureContent } from "../content/features";

export default function FeatureNavigator() {
  const [active, setActive] = useState(featureContent[0].id);
  const activeFeature = featureContent.find((feature) => feature.id === active) || featureContent[0];
  useEffect(() => { const onHash = () => { const id = window.location.hash.replace("#", ""); if (featureContent.some((feature) => feature.id === id)) setActive(id); }; onHash(); window.addEventListener("hashchange", onHash); return () => window.removeEventListener("hashchange", onHash); }, []);
  function selectFeature(id) { setActive(id); window.history.replaceState(null, "", `#${id}`); document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" }); }
  return <div className="feature-navigator" aria-label="Feature navigator"><div className="feature-orbit"><span className="feature-orbit-core">OPTIVOX<br /><b>CORE</b></span><span className="feature-orbit-ring ring-one"></span><span className="feature-orbit-ring ring-two"></span>{featureContent.map((feature, index) => <button type="button" key={feature.id} className={`feature-orbit-node feature-node-${index + 1} ${active === feature.id ? "active" : ""}`} style={{ "--feature-color": feature.color }} onClick={() => selectFeature(feature.id)} aria-label={`Jump to ${feature.title}`}><i></i><span>{feature.title}</span></button>)}</div><div className="feature-navigator-copy"><span className="kicker">Select a capability</span><h3>{activeFeature.title}</h3><p>{activeFeature.plain}</p><div className="feature-status-line"><span style={{ background: activeFeature.color }}></span>{activeFeature.status}<button type="button" className="text-link" onClick={() => selectFeature(activeFeature.id)}>Read the full explanation <span>→</span></button></div></div></div>;
}
