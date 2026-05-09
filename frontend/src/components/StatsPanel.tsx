import type { ActivityEvent, AgentStatus } from '../types';

interface StatsPanelProps {
  events: ActivityEvent[];
  connectionClientCount?: number;
}

export default function StatsPanel({ events }: StatsPanelProps) {
  // Compute statistics from events
  let toolCallCount = 0;
  let toolResultCount = 0;
  let errorCount = 0;
  let currentRound = 0;
  let maxRounds = 0;
  let currentStatus: AgentStatus = 'idle';
  let sessionKey = '';
  const toolUsage: Record<string, number> = {};
  let textChunks = 0;

  for (const event of events) {
    switch (event.type) {
      case 'tool_call':
        toolCallCount++;
        const toolName = String(event.data.tool || 'unknown');
        toolUsage[toolName] = (toolUsage[toolName] || 0) + 1;
        break;
      case 'tool_result':
        toolResultCount++;
        if (event.data.is_error === true) errorCount++;
        break;
      case 'round_progress':
        currentRound = Number(event.data.round || 0);
        maxRounds = Number(event.data.max_rounds || 0);
        break;
      case 'status':
        currentStatus = String(event.data.status || 'idle') as AgentStatus;
        break;
      case 'session_start':
        sessionKey = String(event.data.session_key || '');
        break;
      case 'text_chunk':
        textChunks++;
        break;
      case 'error':
        errorCount++;
        break;
    }
  }

  const topTools = Object.entries(toolUsage)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const statusEmoji: Record<string, string> = {
    idle: '⏸️',
    active: '🔄',
    complete: '✅',
    blocked: '🛑',
    error: '❌',
    max_rounds: '⚠️',
  };

  return (
    <div className="stats-panel">
      <h3>Session Stats</h3>

      <div className="stat-row">
        <span className="stat-label">Status</span>
        <span className="stat-value stat-status">
          {statusEmoji[currentStatus] || '⏸️'} {currentStatus.replace('_', ' ')}
        </span>
      </div>

      {sessionKey && (
        <div className="stat-row">
          <span className="stat-label">Session</span>
          <span className="stat-value stat-session" title={sessionKey}>
            {sessionKey.slice(0, 12)}…
          </span>
        </div>
      )}

      <div className="stat-row">
        <span className="stat-label">Round</span>
        <span className="stat-value">
          {currentRound}{maxRounds ? `/${maxRounds}` : ''}
        </span>
      </div>

      <div className="stat-row">
        <span className="stat-label">Tool Calls</span>
        <span className="stat-value">{toolCallCount}</span>
      </div>

      <div className="stat-row">
        <span className="stat-label">Text Chunks</span>
        <span className="stat-value">{textChunks}</span>
      </div>

      <div className="stat-row">
        <span className="stat-label">Errors</span>
        <span className={`stat-value ${errorCount > 0 ? 'stat-error' : ''}`}>{errorCount}</span>
      </div>

      {topTools.length > 0 && (
        <div className="stat-tools">
          <span className="stat-label">Top Tools</span>
          <div className="tool-usage-list">
            {topTools.map(([name, count]) => (
              <div key={name} className="tool-usage-item">
                <span className="tool-usage-name">{name}</span>
                <span className="tool-usage-count">{count}</span>
                <div className="tool-usage-bar" style={{ width: `${Math.min(100, (count / toolCallCount) * 100)}%` }} />
              </div>
            ))}
          </div>
        </div>
      )}

      {currentRound > 0 && maxRounds > 0 && (
        <div className="stat-progress">
          <div className="progress-bar-container wide">
            <div
              className="progress-bar"
              style={{ width: `${Math.min(100, (currentRound / maxRounds) * 100)}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
