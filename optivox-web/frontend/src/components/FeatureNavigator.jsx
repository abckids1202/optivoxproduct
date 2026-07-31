import { useEffect, useState } from "react";
import { featureContent } from "../content/features";

export default function FeatureNavigator() {
  const [active, setActive] = useState(null);
  const activeFeature = featureContent.find((feature) => feature.id === active) || null;

  useEffect(() => {
    const onHash = () => {
      const id = window.location.hash.replace("#", "");
      if (featureContent.some((feature) => feature.id === id)) setActive(id);
    };
    onHash();
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function selectFeature(id) {
    setActive((current) => current === id ? null : id);
  }

  function showDetails(id) {
    window.history.replaceState(null, "", `#${id}`);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return <div className="feature-navigator" aria-label="Feature navigator">
    <div className="feature-orbit">
      <span className="feature-orbit-core">OPTIVOX<br /><b>CORE</b></span><span className="feature-orbit-ring ring-one"></span><span className="feature-orbit-ring ring-two"></span>
      {featureContent.map((feature, index) => <button type="button" key={feature.id} className={`feature-orbit-node feature-node-${index + 1} ${active === feature.id ? "active" : ""}`} style={{ "--feature-color": feature.color }} onClick={() => selectFeature(feature.id)} aria-label={`Explain ${feature.title}`} aria-pressed={active === feature.id}><i></i><span>{feature.title}</span></button>)}
      {activeFeature && <div className="feature-popover" style={{ "--feature-color": activeFeature.color }} role="status"><button type="button" className="feature-popover-close" onClick={() => setActive(null)} aria-label="Close feature explanation">×</button><span className="tag">{activeFeature.tag}</span><h3>{activeFeature.title}</h3><p>{activeFeature.plain}</p><div className="feature-popover-footer"><span><i></i>{activeFeature.status}</span><button type="button" className="text-link" onClick={() => showDetails(activeFeature.id)}>See full detail <span>→</span></button></div></div>}
    </div>
    <div className="feature-navigator-guide"><span className="kicker">{activeFeature ? "Feature selected" : "Explore the capabilities"}</span><h3>{activeFeature ? "Follow one capability deeper." : "One core, eight connected capabilities."}</h3><p>{activeFeature ? "The selected explanation is temporary. Choose another node to switch it, or close it to return to the map." : "Click any node to reveal a brief explanation. The full feature detail stays below so the map remains easy to scan."}</p></div>
  </div>;
}
