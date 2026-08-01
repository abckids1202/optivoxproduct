import { Loader2 } from "lucide-react";
import Shell from "./components/Shell";
import { useLiveData } from "./hooks/useLiveData";
import Analytics from "./pages/Analytics";
import Attendance from "./pages/Attendance";
import Overview from "./pages/Overview";
import People from "./pages/People";
import Security from "./pages/Security";
import System from "./pages/System";
import { useState } from "react";

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const { state, connection } = useLiveData();

  if (!state) {
    return (
      <div className="boot-screen">
        <Loader2 className="spin" size={28} />
        <strong>Starting Optivox dashboard</strong>
        <span>Preparing local exhibition interface</span>
      </div>
    );
  }

  return (
    <Shell
      activePage={activePage}
      onNavigate={setActivePage}
      connection={connection}
      state={state}
    >
      {activePage === "overview" && <Overview state={state} connection={connection} onNavigate={setActivePage} />}
      {activePage === "attendance" && <Attendance state={state} />}
      {activePage === "security" && <Security state={state} />}
      {activePage === "people" && <People state={state} />}
      {activePage === "analytics" && <Analytics state={state} />}
      {activePage === "system" && <System state={state} connection={connection} />}
    </Shell>
  );
}
