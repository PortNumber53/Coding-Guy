# MCP (Model Context Protocol) Support

Coding Guy now supports MCP servers for extending functionality with external tools. MCP is the same protocol used by Cursor and Windsurf.

## Configuration

Create a config file at `~/.config/coding-guy/config.json`:

```json
{
  "mcpServers": {
    "jira-thing": {
      "command": "/opt/homebrew/bin/npx",
      "args": [
        "mcp-remote",
        "https://jira-thing.truvis.co/mcp?mcp_secret=YOUR_SECRET_HERE"
      ],
      "env": {
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
      "env": {}
    }
  }
}
```

Or copy the example:

```bash
mkdir -p ~/.config/coding-guy
cp .config/coding-guy/config.json.example ~/.config/coding-guy/config.json
# Edit with your actual MCP servers
```

## Usage

When coding-guy starts, it will automatically:
1. Read the config from `~/.config/coding-guy/config.json`
2. Start all configured MCP servers
3. Register their tools with the agent
4. Make them available for use

### Interactive Commands

- Type `mcp` in the chat to see active MCP servers

## MCP Server Discovery

MCP servers can be found in various places:

- **Cursor's MCP servers**: Check Cursor's documentation and extensions
- **Windsurf's MCP servers**: Look in Windsurf's registry
- **Official MCP servers**: https://github.com/modelcontextprotocol
- **Community servers**: Search for "MCP server" on GitHub

## Supported Server Types

Any MCP server that uses stdio transport is supported, such as:

- `mcp-remote` for remote MCP endpoints
- `@modelcontextprotocol/server-filesystem` for filesystem access
- Custom MCP servers written in any language

## Troubleshooting

### MCP servers not loading

Check the console output on startup. You should see messages like:
```
[MCP] Loaded N server configs from ~/.config/coding-guy/config.json
[MCP] Starting server: <name>
[MCP] Server '<name>' started with X tools
```

### MCP tools not appearing

1. Ensure the MCP server started successfully
2. Check that the server supports `tools/list` method
3. Verify the tool definitions are valid JSON

### Server crashes

MCP servers are started as subprocesses. If a server crashes:
- Check server logs (usually printed to stderr)
- Verify the command and args are correct
- Ensure all required environment variables are set
