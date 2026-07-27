import { Camera, Maximize2 } from "lucide-react";
import { FRAME_URL } from "../services/api";

export default function LiveFrame({ engine, connection }) {
  const showImage = connection === "live" && engine?.frameAvailable;
  return (
    <section className="panel live-frame-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Annotated camera</p>
          <h2>{engine?.location || "Class"} feed</h2>
        </div>
        <button className="icon-button" type="button" aria-label="Open full screen">
          <Maximize2 size={18} />
        </button>
      </div>

      <div className="live-frame">
        {showImage ? (
          <img className="frame-image" src={`${FRAME_URL}?t=${Date.now()}`} alt="Latest annotated Optivox frame" />
        ) : (
          <>
            <div className="frame-grid" />
            <div className="scan-line" />
            <div className="frame-empty">
              <Camera size={34} />
              <span>No live frame available</span>
            </div>
          </>
        )}
      </div>

      <div className="frame-meta">
        <span>FPS {engine?.fps || "--"}</span>
        <span>{engine?.camera || "Camera pending"}</span>
        <span>Last frame {engine?.frameAge || "--"}</span>
      </div>
    </section>
  );
}
