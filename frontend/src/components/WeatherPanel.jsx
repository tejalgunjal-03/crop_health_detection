import { useState } from "react";
import { api } from "../api.js";

// Automatic browser location first; manual entry as fallback. Weather is saved to the session on the server.
export default function WeatherPanel({ sessionId, weather, onWeather }) {
  const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  const [manual, setManual] = useState(false); const [q, setQ] = useState("");

  async function fetchWeather(params) {
    setErr(""); setBusy(true);
    try { onWeather(await api(`/api/weather?${new URLSearchParams({ ...params, session_id: sessionId })}`)); }
    catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  function useMyLocation() {
    setErr("");
    if (!navigator.geolocation) { setErr("Your browser does not support location. Please enter it manually."); setManual(true); return; }
    setBusy(true);
    navigator.geolocation.getCurrentPosition(
      (p) => fetchWeather({ lat: p.coords.latitude, lon: p.coords.longitude }),
      (e) => { setBusy(false); setManual(true);
        setErr(e.code === 1 ? "Location permission was denied. Please enter your location manually." : "Could not get your location. Please enter it manually."); },
      { timeout: 15000 });
  }
  return (
    <div className="card">
      <h3>📍 Location & Weather</h3>
      <div className="row">
        <button className="primary" disabled={busy} onClick={useMyLocation}>Use My Current Location</button>
        <button disabled={busy} onClick={() => setManual(!manual)}>Enter Location Manually</button>
      </div>
      {manual && (
        <div className="row">
          <input placeholder="City, Country (e.g. Pune,IN)" value={q} onChange={(e) => setQ(e.target.value)} />
          <button disabled={busy || !q.trim()} onClick={() => fetchWeather({ q: q.trim() })}>Get weather</button>
        </div>
      )}
      {busy && <p className="muted">Loading weather…</p>}
      {err && <div className="error">{err}</div>}
      {weather && (
        <table className="kv"><tbody>
          <tr><td>Location</td><td>{weather.location_name ?? `${weather.latitude?.toFixed(3)}, ${weather.longitude?.toFixed(3)}`}</td></tr>
          <tr><td>Temperature</td><td>{weather.temperature_c ?? "n/a"} °C</td></tr>
          <tr><td>Humidity</td><td>{weather.humidity_pct ?? "n/a"} %</td></tr>
          <tr><td>Rainfall (last hour)</td><td>{weather.rainfall_mm ?? "n/a"} mm</td></tr>
          <tr><td>Wind</td><td>{weather.wind_speed_ms ?? "n/a"} m/s</td></tr>
          <tr><td>Condition</td><td>{weather.description ?? "n/a"}</td></tr>
          <tr><td>Observed at</td><td>{weather.timestamp ?? "n/a"}</td></tr>
        </tbody></table>
      )}
    </div>
  );
}
