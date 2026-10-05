import { useState } from "react";
import { api, auth } from "../api.js";

export default function AuthPage({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [f, setF] = useState({ name: "", email: "", password: "" });
  const [err, setErr] = useState(""); const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  async function submit(e) {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const r = await api(`/api/auth/${mode}`, { method: "POST", json: mode === "login" ? { email: f.email, password: f.password } : f });
      auth.set(r.token, r.user); onLogin(r.user);
    } catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  return (
    <div className="auth card">
      <h1>🌽 Maize Crop Health Monitoring</h1>
      <p className="muted">{mode === "login" ? "Sign in to continue" : "Create your account"}</p>
      <form onSubmit={submit}>
        {mode === "register" && <input placeholder="Your name" value={f.name} onChange={set("name")} required />}
        <input type="email" placeholder="Email" value={f.email} onChange={set("email")} required />
        <input type="password" placeholder="Password (min 6 characters)" value={f.password} onChange={set("password")} required minLength={6} />
        {err && <div className="error">{err}</div>}
        <button className="primary" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Register"}</button>
      </form>
      <button className="link" onClick={() => { setMode(mode === "login" ? "register" : "login"); setErr(""); }}>
        {mode === "login" ? "New here? Create an account" : "Already registered? Sign in"}
      </button>
    </div>
  );
}
