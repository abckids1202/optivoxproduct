import { Loader2 } from "lucide-react";
import Login from "./components/Login";
import Shell from "./components/Shell";
import { useLiveData } from "./hooks/useLiveData";
import Analytics from "./pages/Analytics";
import Attendance from "./pages/Attendance";
import Overview from "./pages/Overview";
import People from "./pages/People";
import Security from "./pages/Security";
import CyberSecurity from "./pages/CyberSecurity";
import System from "./pages/System";
import { logout } from "./services/api";
import { useState } from "react";

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const [authVersion, setAuthVersion] = useState(0);
  const { state, connection, authRequired } = useLiveData(authVersion);

  if (authRequired) {
    return <Login onAuthenticated={() => setAuthVersion((value) => value + 1)} />;
  }

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
      onLogout={async () => {
        await logout();
        setAuthVersion((value) => value + 1);
      }}
    >
      {activePage === "overview" && <Overview state={state} connection={connection} onNavigate={setActivePage} />}
      {activePage === "attendance" && <Attendance state={state} />}
      {activePage === "security" && <Security state={state} />}
      {activePage === "cybersecurity" && <CyberSecurity state={state} />}
      {activePage === "people" && <People state={state} />}
      {activePage === "analytics" && <Analytics state={state} />}
      {activePage === "system" && <System state={state} connection={connection} />}
    </Shell>
  );
}
