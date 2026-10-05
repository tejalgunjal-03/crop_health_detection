import { useEffect, useState } from "react";
import { api } from "../api.js";

const colour = { Healthy: "#2e7d32", "Mild Stress": "#f9a825", "Moderate Stress": "#ef6c00", "Severe/Critical": "#c62828", Inconclusive: "#616161" };

export default function ResultPage({ id }) {
  const [r, setR] = useState(null); const [err, setErr] = useState("");
  useEffect(() => { api(`/api/results/${id}`).then(setR).catch((x) => setErr(x.message)); }, [id]);
  if (err) return <div className="error">{err}</div>;
  if (!r) return <p>Loading…</p>;
  const w = r.evidence.weather; const fm = r.field_monitoring;
  return (
    <>
      <h2>FINAL CROP HEALTH ASSESSMENT</h2>
      <div className="equation">LEAF ANALYSIS <b>+</b> WEATHER CONDITIONS <b>=</b> FINAL CROP HEALTH ASSESSMENT</div>
      <div className="card" style={{ borderLeft: `8px solid ${colour[r.overall_condition] || "#999"}` }}>
        <h3 style={{ color: colour[r.overall_condition] }}>{r.overall_condition}</h3>
        <table className="kv"><tbody>
          <tr><td>Detected disease / issue</td><td>{r.disease}</td></tr>
          <tr><td>Severity</td><td>{r.severity ?? "—"}</td></tr>
          <tr><td>Leaf model confidence</td><td>{(r.confidence * 100).toFixed(1)}%</td></tr>
          <tr><td>Fusion score (0–1)</td><td>{r.fusion_score}</td></tr>
        </tbody></table>
        <h4>Recommendations</h4><ul>{r.recommendations.map((t, i) => <li key={i}>{t}</li>)}</ul>
      </div>
      <div className="grid2">
        <div className="card"><h4>Leaf evidence</h4>
          {Object.entries(r.evidence.leaf.probabilities || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => <div key={k}>{k}: {(v * 100).toFixed(1)}%</div>)}
        </div>
        <div className="card"><h4>Weather context</h4>
          {w ? <>
            <div>{w.location_name} — {w.temperature_c} °C, {w.humidity_pct}% humidity, {w.rainfall_mm} mm rain</div>
            <p>{r.environmental_context}</p>
          </> : <p>{r.environmental_context}</p>}
        </div>
      </div>
      <details className="card"><summary>How was this decided? (rules applied)</summary>
        <ol>{r.evidence.rules_applied.map((t, i) => <li key={i}>{t}</li>)}</ol>
        <p className="muted">{r.disclaimer}</p>
      </details>
      {fm && (
        <div className="card separate">
          <h3>FIELD MONITORING INFORMATION</h3>
          <p className="muted"><b>Separate from the final assessment above.</b> The UAV result only shows where to inspect and was not used in the fusion decision.</p>
          <img className="preview big" src={fm.overlay_url} alt="UAV overlay" />
          <p style={{ marginBottom: "0.5rem" }}>{fm.regions.length} region(s) detected in the field image.</p>
          {fm.regions.length > 0 && (
            <div>
              {fm.regions.map((r) => (
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
        </div>
      )}
    </>
  );
}
