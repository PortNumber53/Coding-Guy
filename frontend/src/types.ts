/**
 * Type definitions for Coding Guy agent activity events
 * exchanged over the WebSocket connection.
 */

export type AgentStatus = 
  | 'complete' 
  | 'max_rounds' 
  | 'error' 
  | 'blocked'
  | 'idle'
  | 'active';

export interface ActivityEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: number;
  meta?: Record<string, unknown>;
}

export interface SessionStartData {
  user_input: string;
  session_key: string;
  max_rounds: number;
}

export interface TextChunkData {
  text: string;
  round: number;
  session_key: string;
}

export interface ToolCallData {
  tool: string;
  arguments: string;
  tool_call_id: string;
  round: number;
  session_key: string;
}

export interface ToolResultData {
  tool: string;
  result: string;
  tool_call_id: string;
  round: number;
  session_key: string;
  is_error: boolean;
}

export interface RoundProgressData {
  round: number;
  max_rounds: number;
  tools_used: string[];
  session_key: string;
}

export interface StatusData {
  status: AgentStatus;
  content?: string;
  round?: number;
  reason?: string;
  question?: string;
  session_key: string;
}

export interface ConnectedData {
  message: string;
  client_count: number;
}

export interface HistoryData {
  events: ActivityEvent[];
  total: number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
