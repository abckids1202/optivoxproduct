import { useEffect, useMemo, useRef, useState } from "react";
import { branchMeta, flowPresets, guidedSequence, mapEdges, mapNodes } from "../data/intelligenceMap";

const nodeById = Object.fromEntries(mapNodes.map((node) => [node.id, node]));

function EdgeLayer({ activeNodes, selectedId, activeFlow }) {
  return <svg className="map-edges" viewBox="0 0 100 100" aria-hidden="true" preserveAspectRatio="none">
    <defs><filter id="edgeGlow"><feGaussianBlur stdDeviation="0.8" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
    {mapEdges.map(([source, target, branch]) => {
      const from = nodeById[source]; const to = nodeById[target]; const focused = activeNodes.has(source) && activeNodes.has(target); const selected = source === selectedId || target === selectedId; const flowColor = activeFlow?.color || branchMeta[branch]?.color || "#46728e";
      return <line key={`${source}-${target}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} className={`map-edge ${focused ? "is-focused" : ""} ${selected ? "is-selected" : ""}`} style={{ "--edge-color": flowColor }} />;
    })}
  </svg>;
}

function NodeButton({ node, active, connected, selected, completed, onSelect }) {
  const meta = branchMeta[node.branch];
  return <button type="button" className={`map-node map-node-${node.kind} ${active ? "is-active" : ""} ${connected ? "is-connected" : ""} ${selected ? "is-selected" : ""} ${completed ? "is-completed" : ""}`} style={{ left: `${node.x}%`, top: `${node.y}%`, "--node-color": meta.color }} onClick={() => onSelect(node.id)} aria-label={`${node.title}: ${node.short}`} aria-pressed={selected}>
    <span className="map-node-orb"><span></span></span><span className="map-node-label">{node.title}</span><span className="map-node-status">{node.status}</span>
  </button>;
}

function DetailPanel({ node, onClose, onFlow }) {
  if (!node) return <aside className="map-detail map-detail-empty"><div><span className="kicker">Select a node</span><h3>Follow the intelligence.</h3><p>Choose any point in the network to see its purpose, inputs, outputs, limits, and related flow.</p></div></aside>;
  const meta = branchMeta[node.branch];
  return <aside className="map-detail" aria-live="polite">
    <div className="map-detail-top"><span className="tag" style={{ color: meta.color, borderColor: `${meta.color}55` }}>{meta.label}</span><button type="button" className="icon-button" onClick={onClose} aria-label="Close node details">×</button></div>
    <h3>{node.title}</h3><p className="map-detail-lead">{node.short}</p><div className="map-detail-status"><span className="status-dot" style={{ background: meta.color }}></span>{node.status}<span className="detail-branch">{meta.label}</span></div>
    <div className="detail-block"><span>Why it exists</span><p>{node.purpose}</p></div><div className="detail-io"><div><span>Input</span><p>{node.input}</p></div><div><span>Process</span><p>{node.process}</p></div><div><span>Output</span><p>{node.output}</p></div></div><div className="detail-block"><span>Technology</span><p className="detail-tech">{node.tech}</p></div><div className="detail-block detail-limit"><span>Limitation</span><p>{node.limitation}</p></div><button type="button" className="btn btn-primary map-follow" onClick={() => onFlow(node.id)}>Follow this path <span>→</span></button>
  </aside>;
}

function MobileWorkflow({ onSelect }) {
  return <div className="mobile-workflow" aria-label="Simple OptiVox workflow">{guidedSequence.map((step, index) => { const node = nodeById[step.nodeId]; return <button type="button" className="mobile-stage" key={step.nodeId} onClick={() => onSelect(step.nodeId)}><span className="mobile-stage-index">0{index + 1}</span><span><strong>{step.title.split(" / ")[1]}</strong><small>{node.short}</small></span><span className="mobile-stage-arrow">→</span></button>; })}</div>;
}

export default function IntelligenceMap() {
  const [selectedId, setSelectedId] = useState("core"); const [branch, setBranch] = useState("all"); const [mode, setMode] = useState("explore"); const [activeFlowId, setActiveFlowId] = useState(null); const [guideIndex, setGuideIndex] = useState(0); const [playing, setPlaying] = useState(false); const [zoom, setZoom] = useState(1); const [pan, setPan] = useState({ x: 0, y: 0 }); const drag = useRef(null);
  const flow = flowPresets.find((preset) => preset.id === activeFlowId) || null; const guideStep = guidedSequence[guideIndex]; const selectedNode = selectedId ? nodeById[selectedId] : null;
  const visibleNodes = useMemo(() => new Set(mapNodes.filter((node) => branch === "all" || node.branch === branch || node.id === "core").map((node) => node.id)), [branch]);
  const activeNodes = useMemo(() => { if (mode === "guided") return new Set(guidedSequence.slice(0, guideIndex + 1).map((step) => step.nodeId).concat("core")); if (flow) return new Set(flow.nodes.concat("core")); return visibleNodes; }, [flow, guideIndex, mode, visibleNodes]);

  useEffect(() => { if (!playing || mode !== "guided") return undefined; const timer = window.setTimeout(() => { setGuideIndex((index) => { if (index >= guidedSequence.length - 1) { setPlaying(false); return index; } return index + 1; }); }, 3600); return () => window.clearTimeout(timer); }, [guideIndex, mode, playing]);
  useEffect(() => { if (mode === "guided" && guideStep) setSelectedId(guideStep.nodeId); }, [guideStep, mode]);
  function selectNode(id) { setSelectedId(id); if (mode === "guided") setPlaying(false); }
  function startFlow(id) { const matching = flowPresets.find((preset) => preset.nodes.includes(id)); if (!matching) return; setMode("explore"); setActiveFlowId(matching.id); setSelectedId(id); }
  function resetView() { setBranch("all"); setActiveFlowId(null); setMode("explore"); setGuideIndex(0); setPlaying(false); setZoom(1); setPan({ x: 0, y: 0 }); setSelectedId("core"); }
  function changeMode(nextMode) { setMode(nextMode); setActiveFlowId(null); if (nextMode === "guided") { setGuideIndex(0); setSelectedId(guidedSequence[0].nodeId); } }
  function onWheel(event) { event.preventDefault(); setZoom((value) => Math.max(0.82, Math.min(1.45, value + (event.deltaY < 0 ? 0.06 : -0.06)))); }
  function onPointerDown(event) { if (event.target.closest("button")) return; drag.current = { x: event.clientX, y: event.clientY, pan }; event.currentTarget.setPointerCapture(event.pointerId); }
  function onPointerMove(event) { if (!drag.current) return; setPan({ x: drag.current.pan.x + (event.clientX - drag.current.x) / 3, y: drag.current.pan.y + (event.clientY - drag.current.y) / 3 }); }
  function onPointerUp() { drag.current = null; }

  return <section className="intelligence-map" aria-labelledby="map-title">
    <div className="map-header"><div><span className="kicker">The Intelligence Map</span><h2 id="map-title">From observation to action.</h2><p>Explore the connected system behind OptiVox. Every node explains what the product sees, remembers, and does with that information.</p></div><div className="map-mode-toggle" role="group" aria-label="Map mode"><button type="button" className={mode === "explore" ? "active" : ""} onClick={() => changeMode("explore")}>Explore</button><button type="button" className={mode === "guided" ? "active" : ""} onClick={() => changeMode("guided")}>Guided tour</button></div></div>
    <div className="map-toolbar"><div className="map-filters"><label htmlFor="branch-filter">Branch</label><select id="branch-filter" value={branch} onChange={(event) => { setBranch(event.target.value); setActiveFlowId(null); }}><option value="all">All systems</option>{Object.entries(branchMeta).filter(([key]) => key !== "core").map(([key, meta]) => <option key={key} value={key}>{meta.label}</option>)}</select><button type="button" className="tool-button" onClick={resetView}>Reset view</button></div><div className="map-zoom" aria-label="Map zoom controls"><button type="button" className="tool-button" onClick={() => setZoom((value) => Math.max(.82, value - .08))} aria-label="Zoom out">−</button><span>{Math.round(zoom * 100)}%</span><button type="button" className="tool-button" onClick={() => setZoom((value) => Math.min(1.45, value + .08))} aria-label="Zoom in">+</button></div></div>
    {mode === "guided" && <div className="guided-bar"><div><span className="kicker">{guideStep.title}</span><p>{guideStep.narration}</p></div><div className="guided-controls"><button type="button" className="tool-button" onClick={() => { setGuideIndex((value) => Math.max(0, value - 1)); setPlaying(false); }}>←</button><button type="button" className="btn btn-primary compact" onClick={() => { if (guideIndex === guidedSequence.length - 1) setGuideIndex(0); setPlaying((value) => !value); }}>{playing ? "Pause" : guideIndex === guidedSequence.length - 1 ? "Replay" : "Play"}</button><button type="button" className="tool-button" onClick={() => { setGuideIndex((value) => Math.min(guidedSequence.length - 1, value + 1)); setPlaying(false); }}>→</button></div></div>}
    <div className="map-layout"><div className="map-canvas-wrap"><div className="map-canvas" onWheel={onWheel} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} role="application" aria-label="Interactive OptiVox system map. Select a node for details."><div className="map-viewport" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}><EdgeLayer activeNodes={activeNodes} selectedId={selectedId} activeFlow={flow} />{mapNodes.map((node) => <NodeButton key={node.id} node={node} active={visibleNodes.has(node.id) && activeNodes.has(node.id)} connected={selectedNode && (node.id === selectedId || mapEdges.some(([source, target]) => (source === selectedId && target === node.id) || (target === selectedId && source === node.id)))} selected={node.id === selectedId} completed={mode === "guided" && activeNodes.has(node.id) && node.id !== guideStep.nodeId} onSelect={selectNode} />)}</div><div className="map-canvas-hint">Drag to pan · scroll to zoom</div></div><div className="map-legend">{Object.entries(branchMeta).filter(([key]) => key !== "core").map(([key, meta]) => <span key={key}><i style={{ background: meta.color }}></i>{meta.label}</span>)}</div></div><DetailPanel node={selectedNode} onClose={() => setSelectedId(null)} onFlow={startFlow} /></div>
    <div className="flow-presets"><div><span className="kicker">Follow a story</span><p>Let one event travel through the system.</p></div>{flowPresets.map((preset) => <button key={preset.id} type="button" className={`flow-chip ${activeFlowId === preset.id ? "active" : ""}`} style={{ "--flow-color": preset.color }} onClick={() => { setActiveFlowId(preset.id); setMode("explore"); setSelectedId(preset.nodes[0]); }}>{preset.label}<span>→</span></button>)}</div>
    <div className="map-mobile-heading"><span className="kicker">Simple workflow</span><p>On a small screen, OptiVox becomes a clear vertical journey.</p></div><MobileWorkflow onSelect={selectNode} />
  </section>;
}
