import { useState } from "react";
import { api } from "../api.js";
import ImagePicker from "../components/ImagePicker.jsx";

export default function UavPage({ session, onContinue }) {
  const [file, setFile] = useState(null); const [res, setRes] = useState(null);
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);

  async function analyze() {
    setErr(""); setRes(null); setBusy(true);
    const form = new FormData(); form.append("session_id", session.session_id); form.append("file", file);
    try { setRes(await api("/api/uav/analyze", { method: "POST", form })); }
    catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  return (
    <>
      <h2>🛰️ UAV Field Monitoring</h2>
      <p className="muted">This step only <b>locates suspicious regions</b>. It is not the final crop-health diagnosis.</p>
      <div className="card">
        <ImagePicker label="1. Choose UAV RGB field image" file={file} onFile={(f) => { setFile(f); setRes(null); }} />
        <button className="primary" disabled={!file || busy} onClick={analyze}>{busy ? "Analysing (large images can take a while)…" : "2. Analyse field image"}</button>
        {err && <div className="error">{err}</div>}
      </div>
      {res && (
        <div className="card">
          <h3>Segmented field map</h3>
          <img className="preview big" src={res.overlay_url} alt="segmentation overlay" />
          <div className="legend">{Object.entries(res.legend).map(([k, c]) => <span key={k}><i style={{ background: `rgb(${c})` }} />{k}</span>)}</div>
          <ul>
            <li>Image size: {res.image_width}×{res.image_height}px, analysed as {res.n_tiles} overlapping 224×224 tiles</li>
            <li>
              Regions detected: <b>{res.regions.length}</b>
              {res.regions.length > 0 && (
                <div style={{ marginTop: "0.6rem" }}>
                  {res.regions.map((r) => (
                    <div key={r.id} style={{ marginBottom: "0.75rem", padding: "0.6rem 0.8rem", background: "rgba(255,255,255,0.04)", borderRadius: "8px", borderLeft: `4px solid rgb(${r.color})` }}>
                      <div style={{ fontWeight: 600, marginBottom: "0.35rem" }}>Region {r.id} — <span style={{ color: `rgb(${r.color})` }}>{r.dominant_label}</span></div>
                      {(r.composition || []).map((c) => (
                        <div key={c.class_name} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.2rem", fontSize: "0.85rem" }}>
                          <span style={{ display: "inline-block", width: 12, height: 12, borderRadius: 3, background: `rgb(${c.color})`, flexShrink: 0 }} />
                          <div style={{ flex: 1, background: "rgba(255,255,255,0.08)", borderRadius: 4, overflow: "hidden", height: 8 }}>
                            <div style={{ width: `${(c.fraction * 100).toFixed(1)}%`, height: "100%", background: `rgb(${c.color})`, transition: "width 0.4s" }} />
                          </div>
                          <span style={{ minWidth: 100 }}>{c.label}</span>
                          <span style={{ fontWeight: 600 }}>{(c.fraction * 100).toFixed(1)}%</span>
                          {c.is_suspicious && <span style={{ fontSize: "0.75rem", color: "#ff7043", fontWeight: 600 }}>⚠ affected</span>}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </li>
            <li>Model-estimated affected pixel share: {(res.suspicious_pixel_fraction * 100).toFixed(1)}% <span className="muted">(estimate only)</span></li>
          </ul>
          <p className="muted">{res.note}</p>
          <div className="notice">👉 <b>{res.next_step}</b></div>
          <button className="primary" onClick={onContinue}>Continue to leaf analysis</button>
        </div>
      )}
    </>
  );
}
