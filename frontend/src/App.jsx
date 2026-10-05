import { useState } from "react";
import { auth } from "./api.js";
import AuthPage from "./pages/AuthPage.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import UavPage from "./pages/UavPage.jsx";
import LeafPage from "./pages/LeafPage.jsx";
import ResultPage from "./pages/ResultPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";

// Simple state-based navigation (no router library) so the flow is easy to explain.
export default function App() {
  const [user, setUser] = useState(auth.user);
  const [page, setPage] = useState("dashboard");   // dashboard | uav | leaf | result | history
  const [session, setSession] = useState(null);     // {session_id, monitoring_mode}
  const [resultId, setResultId] = useState(null);

  const go = (p) => setPage(p);
  if (!user || !auth.token) return <AuthPage onLogin={(u) => { setUser(u); setPage("dashboard"); }} />;

  return (
    <div className="app">
      <header>
        <h1 onClick={() => go("dashboard")}>🌽 Maize Crop Health Monitoring</h1>
        <nav>
          <button onClick={() => go("dashboard")}>Dashboard</button>
          <button onClick={() => go("history")}>History</button>
          <button onClick={() => { auth.clear(); setUser(null); }}>Sign out ({user.name})</button>
        </nav>
      </header>
      <main>
        {page === "dashboard" && <Dashboard onStart={(s) => { setSession(s); go(s.monitoring_mode === "UAV" ? "uav" : "leaf"); }} />}
        {page === "uav" && <UavPage session={session} onContinue={() => go("leaf")} />}
        {page === "leaf" && <LeafPage session={session} onResult={(id) => { setResultId(id); go("result"); }} />}
        {page === "result" && <ResultPage id={resultId} />}
        {page === "history" && <HistoryPage onOpen={(id) => { setResultId(id); go("result"); }} />}
      </main>
    </div>
  );
}
