import { LockKeyhole, LogIn, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { login } from "../services/api";

export default function Login({ onAuthenticated }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result = await login(username.trim(), password);
      onAuthenticated(result.user);
    } catch (requestError) {
      setError(requestError.message || "Sign-in failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-screen">
      <section className="auth-panel" aria-labelledby="login-title">
        <div className="auth-brand"><span><ShieldCheck size={22} /></span><div><strong>Optivox</strong><small>Local school intelligence</small></div></div>
        <p className="eyebrow">Protected operator console</p>
        <h1 id="login-title">Sign in to Optivox</h1>
        <p className="auth-copy">Use an authorized operator account to view live camera, attendance, and security data.</p>
        <form onSubmit={submit} className="auth-form">
          <label><span>Username</span><input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
          <label><span>Password</span><div className="password-field"><LockKeyhole size={16} /><input type="password" autoComplete="current-password" minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} required /></div></label>
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button type="submit" disabled={submitting}>{submitting ? "Signing in..." : <><LogIn size={17} /> Sign in</>}</button>
        </form>
        <p className="auth-note">Biometric material stays with the local edge engine.</p>
      </section>
    </main>
  );
}
