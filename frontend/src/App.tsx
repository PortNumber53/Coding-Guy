import { useState, useRef, useEffect, useMemo } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import ActivityFeed from './components/ActivityFeed';
import ConnectionIndicator from './components/ConnectionIndicator';
import StatsPanel from './components/StatsPanel';
import type { ActivityEvent } from './types';
import './App.css';

// Default WebSocket URL - can be overridden via env or URL param
const getWsUrl = () => {
  const params = new URLSearchParams(window.location.search);
  const fromParam = params.get('ws');
  if (fromParam) return fromParam;
  const fromEnv = import.meta.env.VITE_WS_URL;
  if (fromEnv) return fromEnv as string;
  // Default: same host, port 8765
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.hostname}:8765`;
};

function App() {
  const [wsUrl, setWsUrl] = useState(getWsUrl);
  const [urlInput, setUrlInput] = useState(getWsUrl());
  const [autoScroll, setAutoScroll] = useState(true);
  const [showStats, setShowStats] = useState(true);
  const feedRef = useRef<HTMLDivElement>(null);

  const { events, connectionStatus, clearEvents } = useWebSocket(wsUrl);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && feedRef.current) {
      const el = feedRef.current.querySelector('.feed-events');
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, [events, autoScroll]);

  // Compute accumulated text from text_chunk events
  const accumulatedText = useMemo(() => {
    return events
      .filter((e: ActivityEvent) => e.type === 'text_chunk')
      .map((e: ActivityEvent) => String(e.data.text || ''))
      .join('');
  }, [events]);

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setWsUrl(urlInput);
  };

  const handleReconnect = () => {
    // Force reconnection by toggling the URL
    setWsUrl('');
    setTimeout(() => setWsUrl(urlInput), 100);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>🤖 Coding Guy</h1>
          <span className="header-subtitle">Agent Activity Monitor</span>
        </div>
        <div className="header-right">
          <ConnectionIndicator status={connectionStatus} onReconnect={handleReconnect} />
          <form className="url-form" onSubmit={handleUrlSubmit}>
            <input
              type="text"
              className="url-input"
              value={urlInput}
              onChange={e => setUrlInput(e.target.value)}
              placeholder="ws://localhost:8765"
              title="WebSocket server URL"
            />
            <button type="submit" className="btn-connect">Connect</button>
          </form>
        </div>
      </header>

      <div className="app-body">
        <aside className={`sidebar ${showStats ? '' : 'sidebar-collapsed'}`}>
          <div className="sidebar-toggle" onClick={() => setShowStats(!showStats)}>
            {showStats ? '◀' : '▶'} Stats
          </div>
          {showStats && <StatsPanel events={events} />}
        </aside>

        <main className="main-content" ref={feedRef}>
          <div className="feed-toolbar">
            <label className="auto-scroll-toggle">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={e => setAutoScroll(e.target.checked)}
              />
              Auto-scroll
            </label>
            {accumulatedText && (
              <div className="accumulated-text">
                <strong>Agent Output:</strong>
                <pre className="output-preview">{accumulatedText.slice(-500)}</pre>
              </div>
            )}
          </div>

          <ActivityFeed events={events} onClear={clearEvents} />
        </main>
      </div>

      <footer className="app-footer">
        <span>Coding Guy Agent Dashboard</span>
        <span className="footer-separator">·</span>
        <span>WebSocket: {wsUrl.replace(/^wss?:\/\//, '')}</span>
        <span className="footer-separator">·</span>
        <span>Events: {events.length}</span>
      </footer>
    </div>
  );
}

export default App;
