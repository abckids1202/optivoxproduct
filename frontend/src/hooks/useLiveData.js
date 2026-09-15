import { useEffect, useState } from "react";
import { fetchDashboardState } from "../services/api";

export function useLiveData(reloadKey = 0) {
  const [state, setState] = useState(null);
  const [connection, setConnection] = useState("connecting");
  const [authRequired, setAuthRequired] = useState(false);

  useEffect(() => {
    let active = true;
    let intervalId;

    async function load() {
      try {
        const data = await fetchDashboardState();
        if (!active) return;
        setState(data);
        setAuthRequired(Boolean(data?.authRequired));
        if (data?.dataMode === "auth_required") setConnection("auth_required");
        else if (data?.dataMode === "demo") setConnection("demo");
        else if (data?.dataMode === "backend_offline") setConnection("backend_offline");
        else if (data?.engine?.status === "Offline") setConnection("engine_offline");
        else setConnection("live");
      } catch {
        if (active) setConnection("backend_offline");
      }
    }

    load();
    intervalId = window.setInterval(load, 2500);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [reloadKey]);

  return { state, connection, authRequired };
}
