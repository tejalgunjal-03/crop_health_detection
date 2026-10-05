import { useState } from "react";
import { api } from "../api.js";

export default function Dashboard({ onStart }) {
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  async function start(mode) {
    setErr(""); setBusy(true);
    try { onStart(await api("/api/sessions", { method: "POST", json: { monitoring_mode: mode } })); }
    catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  return (
    <>
      <h2>Choose Monitoring Method</h2>
      {err && <div className="error">{err}</div>}
      <div className="grid2">
        <div className="card">
          <h3>🛰️ UAV Field Monitoring</h3>
          <p>Use for a <b>larger field</b>. Upload a UAV (drone) photo and the system highlights suspicious or affected regions so you know where to inspect.</p>
          <p className="muted">Then you upload a close-up leaf image from the highlighted area for the final assessment.</p>
          <button className="primary" disabled={busy} onClick={() => start("UAV")}>Start UAV Field Monitoring</button>
        </div>
        <div className="card">
          <h3>🍃 Direct Leaf Analysis</h3>
          <p>Use for a <b>small field</b>. Upload a close-up photo of a maize leaf directly. No UAV image is needed.</p>
          <p className="muted">The final assessment uses the leaf image plus local weather.</p>
          <button className="primary" disabled={busy} onClick={() => start("DIRECT_LEAF")}>Start Direct Leaf Analysis</button>
        </div>
      </div>
    </>
  );
}
