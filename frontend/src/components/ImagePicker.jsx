import { useEffect, useState } from "react";

// File chooser with preview. Checks type/size in the browser for fast feedback (the server re-checks).
export default function ImagePicker({ label, file, onFile, maxMB = 15 }) {
  const [url, setUrl] = useState(null); const [err, setErr] = useState("");
  useEffect(() => { if (!file) { setUrl(null); return; } const u = URL.createObjectURL(file); setUrl(u); return () => URL.revokeObjectURL(u); }, [file]);
  function pick(e) {
    const f = e.target.files[0]; setErr("");
    if (!f) return;
    if (!["image/jpeg", "image/png"].includes(f.type)) { setErr("Please choose a JPG or PNG image."); onFile(null); return; }
    if (f.size > maxMB * 1024 * 1024) { setErr(`Image is too large (max ${maxMB} MB).`); onFile(null); return; }
    onFile(f);
  }
  return (
    <div>
      <label className="filebtn">{label}<input type="file" accept="image/jpeg,image/png" onChange={pick} hidden /></label>
      {err && <div className="error">{err}</div>}
      {url && <img className="preview" src={url} alt="preview" />}
    </div>
  );
}
