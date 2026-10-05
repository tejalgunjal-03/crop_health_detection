// Small fetch wrapper. Every failure becomes a plain, user-friendly Error message.
export const auth = {
  get token() { return localStorage.getItem("token"); },
  set(token, user) { localStorage.setItem("token", token); localStorage.setItem("user", JSON.stringify(user)); },
  clear() { localStorage.removeItem("token"); localStorage.removeItem("user"); },
  get user() { try { return JSON.parse(localStorage.getItem("user")); } catch { return null; } },
};

export async function api(path, { method = "GET", json, form } = {}) {
  const headers = {};
  if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
  let body;
  if (json) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  let res;
  try {
    res = await fetch(path, { method, headers, body });
  } catch {
    throw new Error("Cannot reach the server. Please check that the backend is running and your internet is working.");
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    if (res.status === 401) auth.clear();
    const msg = data?.error?.message || (data?.detail ? "Please check the information you entered." : null);
    const err = new Error(msg || "Something went wrong. Please try again.");
    err.code = data?.error?.code; err.status = res.status;
    throw err;
  }
  if (data === null) throw new Error("The server returned an empty result.");
  return data;
}
