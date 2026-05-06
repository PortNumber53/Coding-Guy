# Coding Guy — Frontend

Real-time agent activity monitor dashboard built with React + Vite on Cloudflare Workers.

## Features

- **Live Activity Feed** — Watch agent activity in real-time via WebSocket
- **Tool Call Tracking** — See every tool call, its arguments, and results
- **Progress Monitoring** — Track conversation rounds, progress bars
- **Session Stats** — Tool usage, error counts, round progress
- **Dark Theme** — Optimized for long monitoring sessions
- **Auto-reconnect** — Reconnects to WebSocket on disconnect

## Getting Started

### 1. Start the backend with WebSocket support

```bash
cd ..
python3 coding_agent.py --ws --ws-port 8765 [other flags...]
```

The `--ws` flag starts a WebSocket server (default port 8765) that broadcasts agent activity events.

### 2. Start the frontend dev server

```bash
npm install
npm run dev
```

Open http://localhost:5173 — the dashboard will automatically connect to the WebSocket server.

### Configuration

- **WebSocket URL**: Override with `?ws=ws://your-host:8765` query parameter or `VITE_WS_URL` env variable
- **Default**: Auto-detects from current hostname on port 8765

### Build for Production

```bash
npm run build
```

### Deploy to Cloudflare Workers

```bash
npm run deploy
```

## Architecture

```
┌──────────────┐          WebSocket          ┌──────────────┐
│   Frontend   │ ←──────────────────────────→ │  ws_server   │
│  React + Vite │   activity events stream    │  (Python)    │
│              │                              │              │
│  - ActivityFeed                             │  - ActivityBroadcaster
│  - StatsPanel                              │  - Client management
│  - ConnectionIndicator                     │  - Event history
└──────────────┘                              └──────┬───────┘
                                                     │
                                              activity_callback
                                                     │
                                              ┌──────▼───────┐
                                              │ coding_agent  │
                                              │  (agent_loop) │
                                              └──────────────┘
```

## Event Types

| Event | Description |
|-------|-------------|
| `session_start` | Agent begins processing a user's task |
| `text_chunk` | Streaming text delta from LLM response |
| `tool_call` | Agent calls a tool (with arguments) |
| `tool_result` | Tool execution result (with output) |
| `round_progress` | Round progress with tools used |
| `status` | Agent status change (complete, error, blocked) |
| `error` | Error occurred during processing |
| `connected` | Client connected to WebSocket server |
| `history` | Replayed event history on connect |
