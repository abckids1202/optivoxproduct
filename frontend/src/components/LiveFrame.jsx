import { Camera, Maximize2 } from "lucide-react";
import { useEffect, useState } from "react";
import { FRAME_URL, requestHeaders } from "../services/api";

export default function LiveFrame({ engine, connection }) {
  const showImage = connection === "live" && engine?.frameAvailable;
  const [frameUrl, setFrameUrl] = useState("");

  useEffect(() => {
    let active = true;
    let objectUrl = "";
    let timer;

    if (!showImage) {
      setFrameUrl("");
      return () => {};
    }

    const loadFrame = async () => {
      try {
        const response = await fetch(`${FRAME_URL}?t=${Date.now()}`, { headers: requestHeaders() });
        if (!response.ok) throw new Error("Frame unavailable");
        const blob = await response.blob();
        if (!active) return;
        const nextObjectUrl = URL.createObjectURL(blob);
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = nextObjectUrl;
        setFrameUrl(nextObjectUrl);
      } catch {
        if (active) setFrameUrl("");
      } finally {
        if (active) timer = window.setTimeout(loadFrame, 500);
      }
    };

    loadFrame();
    return () => {
      active = false;
      window.clearTimeout(timer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [showImage]);

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
        {showImage && frameUrl ? (
          <img className="frame-image" src={frameUrl} alt="Latest annotated Optivox frame" />
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
