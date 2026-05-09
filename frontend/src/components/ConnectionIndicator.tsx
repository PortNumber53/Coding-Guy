import type { ConnectionStatus } from '../types';

interface ConnectionIndicatorProps {
  status: ConnectionStatus;
  onReconnect?: () => void;
}

const statusConfig: Record<ConnectionStatus, { label: string; color: string; pulse: boolean }> = {
  connected: { label: 'Connected', color: '#22c55e', pulse: true },
  connecting: { label: 'Connecting', color: '#eab308', pulse: true },
  disconnected: { label: 'Disconnected', color: '#ef4444', pulse: false },
  error: { label: 'Error', color: '#ef4444', pulse: false },
};

export default function ConnectionIndicator({ status, onReconnect }: ConnectionIndicatorProps) {
  const config = statusConfig[status];

  return (
    <div className="connection-indicator">
      <span
        className={`status-dot ${config.pulse ? 'pulse' : ''}`}
        style={{ backgroundColor: config.color }}
      />
      <span className="status-label">{config.label}</span>
      {(status === 'disconnected' || status === 'error') && onReconnect && (
        <button className="btn-reconnect" onClick={onReconnect}>
          Reconnect
        </button>
      )}
    </div>
  );
}
