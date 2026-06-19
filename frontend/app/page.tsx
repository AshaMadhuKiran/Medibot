"use client";

import { useState, useRef, useEffect } from "react";
import { chat, login, ChatResponse } from "@/lib/api";
import { DEMO_ACCOUNTS, ROLE_COLORS } from "@/lib/demoUsers";

interface Session {
  token: string;
  role: string;
  displayName: string;
  collections: string[];
}

interface Message {
  role: "user" | "bot";
  text: string;
  data?: ChatResponse;
}

function prettyRole(role: string) {
  return role.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  return (
    <main className="app">
      {session ? (
        <ChatView session={session} onLogout={() => setSession(null)} />
      ) : (
        <LoginView onLogin={setSession} />
      )}
    </main>
  );
}

function LoginView({ onLogin }: { onLogin: (s: Session) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(u: string, p: string) {
    setError("");
    setBusy(true);
    try {
      const res = await login(u, p);
      onLogin({
        token: res.token,
        role: res.role,
        displayName: res.display_name,
        collections: res.collections,
      });
    } catch (e: any) {
      setError(e.message || "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <div className="login-card">
        <div className="brand">
          <span className="logo">⚕️</span>
          <div>
            <h1>MediBot</h1>
            <p>MediAssist Health Network — secure internal assistant</p>
          </div>
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(username, password);
          }}
        >
          <label>Username</label>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. nurse.priya"
            autoFocus
          />
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="medibot123"
          />
          {error && <div className="error">{error}</div>}
          <button type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div className="demo">
          <p>Demo accounts (click to log in):</p>
          <div className="demo-grid">
            {DEMO_ACCOUNTS.map((a) => (
              <button
                key={a.username}
                className="demo-btn"
                style={{ borderColor: ROLE_COLORS[a.role] }}
                onClick={() => {
                  setUsername(a.username);
                  setPassword(a.password);
                  submit(a.username, a.password);
                }}
                disabled={busy}
              >
                <span
                  className="dot"
                  style={{ background: ROLE_COLORS[a.role] }}
                />
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatView({
  session,
  onLogout,
}: {
  session: Session;
  onLogout: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const res = await chat(session.token, q);
      setMessages((m) => [...m, { role: "bot", text: res.answer, data: res }]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "bot", text: `Error: ${e.message}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  const roleColor = ROLE_COLORS[session.role] || "#555";

  return (
    <div className="chat-layout">
      <aside className="sidebar">
        <div className="brand small">
          <span className="logo">⚕️</span>
          <h2>MediBot</h2>
        </div>

        <div className="who">
          <div className="avatar" style={{ background: roleColor }}>
            {session.displayName.charAt(0)}
          </div>
          <div>
            <div className="name">{session.displayName}</div>
            <span className="role-badge" style={{ background: roleColor }}>
              {prettyRole(session.role)}
            </span>
          </div>
        </div>

        <div className="collections">
          <h3>Accessible collections</h3>
          <ul>
            {session.collections.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <p className="hint">
            Queries are filtered to these collections at the database level.
          </p>
        </div>

        <button className="logout" onClick={onLogout}>
          Log out
        </button>
      </aside>

      <section className="chat-main">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h3>Ask MediBot a question</h3>
              <p>
                Try clinical protocols, nursing procedures, billing rules, or
                analytical questions like “how many claims were escalated?”
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          {busy && <div className="bubble bot typing">MediBot is thinking…</div>}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask a question…"
            disabled={busy}
          />
          <button onClick={send} disabled={busy || !input.trim()}>
            Send
          </button>
        </div>
      </section>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return <div className="bubble user">{message.text}</div>;
  }

  const data = message.data;
  const blocked = data?.access_blocked;
  const tag =
    data?.retrieval_type === "sql_rag"
      ? "SQL RAG"
      : data?.retrieval_type === "hybrid_rag"
      ? "Hybrid RAG"
      : data?.retrieval_type === "rbac_blocked"
      ? "Access Blocked"
      : null;

  return (
    <div className={`bubble bot ${blocked ? "blocked" : ""}`}>
      {tag && (
        <span className={`tag ${data?.retrieval_type}`}>{tag}</span>
      )}
      <div className="answer">{message.text}</div>

      {data?.sql && (
        <details className="sql">
          <summary>Generated SQL</summary>
          <pre>{data.sql}</pre>
        </details>
      )}

      {data?.sources && data.sources.length > 0 && (
        <div className="sources">
          <div className="sources-title">Sources</div>
          {data.sources.map((s, i) => (
            <div className="source" key={i}>
              <span className="src-doc">{s.source_document}</span>
              <span className="src-sec">{s.section_title}</span>
              <span className="src-coll">{s.collection}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
