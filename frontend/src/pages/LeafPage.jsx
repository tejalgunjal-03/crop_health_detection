import { useState } from "react";
import { api } from "../api.js";
import ImagePicker from "../components/ImagePicker.jsx";
import WeatherPanel from "../components/WeatherPanel.jsx";

export default function LeafPage({ session, onResult }) {
  const [file, setFile] = useState(null); const [leaf, setLeaf] = useState(null); const [weather, setWeather] = useState(null);
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [fusing, setFusing] = useState(false);

  async function analyze() {
    setErr(""); setLeaf(null); setBusy(true);
    const form = new FormData(); form.append("session_id", session.session_id); form.append("file", file);
    try { setLeaf(await api("/api/leaf/analyze", { method: "POST", form })); }
    catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  async function fuse() {
    setErr(""); setFusing(true);
    try { const r = await api("/api/fusion/analyze", { method: "POST", json: { session_id: session.session_id } }); onResult(r.id); }
    catch (x) { setErr(x.message); setFusing(false); }
  }
  return (
    <>
      <h2>🍃 Leaf Analysis</h2>
      {session.monitoring_mode === "UAV" && <div className="notice">Upload a close-up leaf from the region highlighted in the UAV map.</div>}
      <div className="grid2">
        <div className="card">
          <ImagePicker label="1. Choose close-up maize leaf image" file={file} onFile={(f) => { setFile(f); setLeaf(null); }} />
          <button className="primary" disabled={!file || busy} onClick={analyze}>{busy ? "Analysing…" : "2. Analyse leaf"}</button>
          {leaf && (
            <div className="result">
              <h3>{leaf.disease} <span className="muted">({(leaf.confidence * 100).toFixed(1)}% confidence)</span></h3>
              {Object.entries(leaf.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                <div key={k} className="bar"><span>{k}</span><div><i style={{ width: `${v * 100}%` }} /></div><b>{(v * 100).toFixed(1)}%</b></div>
              ))}
            </div>
          )}
        </div>
        <WeatherPanel sessionId={session.session_id} weather={weather} onWeather={setWeather} />
      </div>
      {err && <div className="error">{err}</div>}
      <div className="card">
        <b>LEAF ANALYSIS + WEATHER CONDITIONS → FINAL ASSESSMENT</b>
        {leaf && !weather && <p className="muted">No weather loaded yet. You can continue, but the assessment will use the leaf evidence only.</p>}
        <div><button className="primary" disabled={!leaf || fusing} onClick={fuse}>{fusing ? "Working…" : "3. Get final crop health assessment"}</button></div>
      </div>
    </>
  );
}
