import type { ActivityEvent } from '../types';

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatRelativeTime(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 5) return 'just now';
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return formatTime(ts);
}

interface EventIconProps {
  type: string;
}

function EventIcon({ type }: EventIconProps) {
  const iconMap: Record<string, string> = {
    session_start: '🚀',
    text_chunk: '💬',
    tool_call: '🔧',
    tool_result: '📋',
    round_progress: '🔄',
    status: '⚡',
    error: '❌',
    connected: '🔗',
    history: '📜',
    pong: '📶',
  };
  return <span className="event-icon">{String(iconMap[type] ?? '📡')}</span>;
}

interface EventBadgeProps {
  type: string;
}

function EventBadge({ type }: EventBadgeProps) {
  const colorMap: Record<string, string> = {
    session_start: 'badge-start',
    text_chunk: 'badge-text',
    tool_call: 'badge-tool',
    tool_result: 'badge-result',
    round_progress: 'badge-progress',
    status: 'badge-status',
    error: 'badge-error',
    connected: 'badge-connected',
  };
  const cls = colorMap[type] || 'badge-default';
  return <span className={`event-badge ${cls}`}>{String(type)}</span>;
}

function tryParseArgs(argsStr: string): Record<string, unknown> | null {
  try {
    return JSON.parse(argsStr);
  } catch {
    return null;
  }
}

function truncateText(text: string, maxLen: number = 300): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '…';
}

interface ActivityEventItemProps {
  event: ActivityEvent;
}

function ActivityEventItem({ event }: ActivityEventItemProps) {
  const { type, data, timestamp } = event;

  const renderContent = () => {
    switch (type) {
      case 'session_start':
        return (
          <div className="event-content">
            <strong>New Session</strong>
            <p className="event-input">{truncateText(String(data.user_input || ''), 200)}</p>
            <span className="event-meta">Max rounds: {String(data.max_rounds || '?')}</span>
          </div>
        );

      case 'text_chunk':
        return (
          <div className="event-content event-text-chunk">
            <span className="chunk-text">{String(data.text || '')}</span>
          </div>
        );

      case 'tool_call': {
        const parsed = tryParseArgs(String(data.arguments || '{}'));
        return (
          <div className="event-content">
            <strong className="tool-name">{String(data.tool || 'unknown')}</strong>
            {parsed ? (
              <pre className="tool-args">{JSON.stringify(parsed, null, 2)}</pre>
            ) : (
              <code className="tool-args-raw">{truncateText(String(data.arguments || ''))}</code>
            )}
          </div>
        );
      }

      case 'tool_result': {
        const isError = data.is_error === true;
        const resultText = String(data.result || '');
        return (
          <div className={`event-content ${isError ? 'result-error' : 'result-ok'}`}>
            <strong>{String(data.tool || 'unknown')}</strong>
            <pre className="tool-result">{truncateText(resultText, 500)}</pre>
          </div>
        );
      }

      case 'round_progress': {
        const round = Number(data.round || 0);
        const maxRounds = Number(data.max_rounds || 1);
        const pct = Math.min(100, (round / maxRounds) * 100);
        const tools = (data.tools_used as string[]) || [];
        return (
          <div className="event-content">
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${pct}%` }} />
              <span className="progress-label">{round}/{maxRounds}</span>
            </div>
            <div className="tools-used">
              {tools.map(t => <span key={t} className="tool-tag">{t}</span>)}
            </div>
          </div>
        );
      }

      case 'status': {
        const status = String(data.status || '');
        const statusEmoji: Record<string, string> = {
          complete: '✅',
          max_rounds: '⚠️',
          error: '❌',
          blocked: '🛑',
        };
        return (
          <div className="event-content">
            <strong>{String(statusEmoji[status] ?? '⚡')} {status.replace('_', ' ')}</strong>
            {typeof data.content === 'string' && data.content && <p className="status-content">{truncateText(data.content, 300)}</p>}
            {typeof data.question === 'string' && data.question && <p className="status-question">❓ {data.question}</p>}
          </div>
        );
      }

      case 'error':
        return (
          <div className="event-content result-error">
            <strong>Error</strong>
            <p>{truncateText(String(data.message || 'Unknown error'), 300)}</p>
          </div>
        );

      case 'connected':
        return (
          <div className="event-content">
            <span>{String(data.message || 'Connected')}</span>
            <span className="event-meta">Clients: {String(data.client_count || 0)}</span>
          </div>
        );

      default:
        return (
          <div className="event-content">
            <pre className="raw-data">{JSON.stringify(data, null, 2)}</pre>
          </div>
        );
    }
  };

  return (
    <div className={`activity-event event-type-${type}`}>
      <div className="event-header">
        <EventIcon type={type} />
        <EventBadge type={type} />
        <span className="event-time" title={formatTime(timestamp)}>{formatRelativeTime(timestamp)}</span>
      </div>
      {renderContent()}
    </div>
  );
}

interface ActivityFeedProps {
  events: ActivityEvent[];
  onClear: () => void;
}

export default function ActivityFeed({ events, onClear }: ActivityFeedProps) {
  return (
    <div className="activity-feed">
      <div className="feed-header">
        <h2>Agent Activity</h2>
        <div className="feed-controls">
          <span className="event-count">{events.length} events</span>
          <button className="btn-clear" onClick={onClear} title="Clear events">
            Clear
          </button>
        </div>
      </div>
      <div className="feed-events">
        {events.length === 0 ? (
          <div className="feed-empty">
            <p>Waiting for agent activity…</p>
            <p className="feed-empty-hint">Start the backend with <code>--ws</code> flag to enable the WebSocket feed.</p>
          </div>
        ) : (
          events.map((event, idx) => (
            <ActivityEventItem key={`${event.timestamp}-${idx}`} event={event} />
          ))
        )}
      </div>
    </div>
  );
}
