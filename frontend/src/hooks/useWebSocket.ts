import { useEffect, useRef, useState, useCallback } from 'react';
import type { ActivityEvent, ConnectionStatus } from '../types';

const WS_RECONNECT_DELAY = 2000;
const WS_PING_INTERVAL = 30000;
const MAX_EVENTS = 1000;

interface UseWebSocketReturn {
  events: ActivityEvent[];
  connectionStatus: ConnectionStatus;
  clearEvents: () => void;
  send: (data: unknown) => void;
}

export function useWebSocket(url: string): UseWebSocketReturn {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined!);
  const pingTimerRef = useRef<ReturnType<typeof setInterval>>(undefined!);
  const mountedRef = useRef(true);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return;

    setConnectionStatus('connecting');

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus('connected');
        console.log('[WS] Connected to', url);

        // Start ping interval
        pingTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, WS_PING_INTERVAL);
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const parsed: ActivityEvent = JSON.parse(event.data);

          if (parsed.type === 'history') {
            // Load history events
            const historyEvents = (parsed.data as { events: ActivityEvent[] }).events || [];
            setEvents(prev => {
              // Merge history at the beginning, then any existing events
              const existingTimestamps = new Set(prev.map(e => e.timestamp));
              const newFromHistory = historyEvents.filter(e => !existingTimestamps.has(e.timestamp));
              const merged = [...newFromHistory, ...prev];
              return merged.slice(-MAX_EVENTS);
            });
            return;
          }

          if (parsed.type === 'pong') return;

          setEvents(prev => {
            const next = [...prev, parsed];
            return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
          });
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionStatus('disconnected');
        console.log('[WS] Disconnected');
        clearInterval(pingTimerRef.current);

        // Auto-reconnect
        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, WS_RECONNECT_DELAY);
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setConnectionStatus('error');
      };

      wsRef.current = ws;
    } catch (e) {
      setConnectionStatus('error');
      console.error('[WS] Connection error:', e);
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, WS_RECONNECT_DELAY);
    }
  }, [url]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      clearInterval(pingTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { events, connectionStatus, clearEvents, send };
}
