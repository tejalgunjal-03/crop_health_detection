import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function HistoryPage({ onOpen }) {
  const [rows, setRows] = useState(null); const [err, setErr] = useState("");
  useEffect(() => { api("/api/results/history").then(setRows).catch((x) => setErr(x.message)); }, []);
  if (err) return <div className="error">{err}</div>;
  if (!rows) return <p>Loading…</p>;
  return (
    <>
      <h2>History / Reports</h2>
      {rows.length === 0 ? <p className="muted">No assessments yet.</p> : (
        <table className="list"><thead><tr><th>Date</th><th>Mode</th><th>Condition</th><th>Disease</th><th>Confidence</th><th /></tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.id}><td>{new Date(r.created_at).toLocaleString()}</td><td>{r.monitoring_mode === "UAV" ? "UAV" : "Direct leaf"}</td>
              <td>{r.overall_condition}</td><td>{r.disease}</td><td>{(r.confidence * 100).toFixed(0)}%</td>
              <td><button onClick={() => onOpen(r.id)}>Open</button></td></tr>))}</tbody></table>
      )}
    </>
  );
}
