"""MCP (Model Context Protocol) Client for Coding Guy.

Reads MCP server configuration from ~/.config/coding-guy/config.json
using the same format as Cursor and Windsurf.
"""

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class MCPServer:
    """Manages connection to a single MCP server process."""
    
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start the MCP server subprocess."""
        try:
            # Merge with current environment
            env = os.environ.copy()
            env.update(self.env)
            
            cmd = [self.command] + self.args
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1
            )
            
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "coding-guy",
                        "version": "1.0.0"
                    }
                }
            }
            
            line = json.dumps(init_request) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                print(f"[MCP] No response from server '{self.name}'", file=sys.stderr)
                return False
            
            response = json.loads(response_line)
            if "error" in response:
                print(f"[MCP] Init error for '{self.name}': {response['error']}", file=sys.stderr)
                return False
            
            print(f"[MCP] Server '{self.name}' started successfully", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"[MCP] Failed to start server '{self.name}': {e}", file=sys.stderr)
            return False
    
    def stop(self):
        """Stop the MCP server subprocess."""
        with self._lock:
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                except Exception as e:
                    print(f"[MCP] Error stopping server: {e}", file=sys.stderr)
                finally:
                    self.process = None
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from this MCP server."""
        if not self.process or not self.process.stdin:
            return []
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/list",
                "params": {}
            }
            
            line = json.dumps(request) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                return []
            
            response = json.loads(response_line)
            if "error" in response:
                return []
            
            result = response.get("result", {})
            return result.get("tools", [])
            
        except Exception as e:
            print(f"[MCP] Error listing tools: {e}", file=sys.stderr)
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on this MCP server."""
        if not self.process or not self.process.stdin:
            return {"error": "Server not running"}
        
        with self._lock:
            try:
                request = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
                
                line = json.dumps(request) + "\n"
                self.process.stdin.write(line)
                self.process.stdin.flush()
                
                response_line = self.process.stdout.readline()
                if not response_line:
                    return {"error": "No response from server"}
                
                response = json.loads(response_line)
                if "error" in response:
                    return {"error": response["error"]}
                
                result = response.get("result", {})
                return result
                
            except Exception as e:
                return {"error": f"Tool call failed: {str(e)}"}


class MCPClient:
    """Client for managing multiple MCP servers."""
    
    CONFIG_PATH = Path.home() / ".config" / "coding-guy" / "config.json"
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
    
    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP server configuration from file."""
        # Allow override via environment variable
        config_path = Path(os.getenv("CODING_GUY_CONFIG_PATH", self.CONFIG_PATH))
        
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            
            # Support both direct server configs and mcpServers wrapper
            mcp_servers = data.get("mcpServers", data)
            
            print(f"[MCP] Loaded config from {config_path}", file=sys.stderr)
            return mcp_servers
            
        except json.JSONDecodeError as e:
            print(f"[MCP] Invalid JSON in config: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[MCP] Failed to load config: {e}", file=sys.stderr)
        
        return {}
    
    def start_servers(self):
        """Start all MCP servers from config."""
        config = self.load_config()
        
        for name, server_config in config.items():
            command = server_config.get("command", "")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            
            if not command:
                print(f"[MCP] Server '{name}' has no command, skipping", file=sys.stderr)
                continue
            
            print(f"[MCP] Starting server: {name}", file=sys.stderr)
            
            server = MCPServer(name=name, command=command, args=args, env=env)
            if server.start():
                self.servers[name] = server
            else:
                print(f"[MCP] Failed to start server: {name}", file=sys.stderr)
    
    def stop_servers(self):
        """Stop all MCP servers."""
        for name, server in self.servers.items():
            print(f"[MCP] Stopping server: {name}", file=sys.stderr)
            server.stop()
        self.servers.clear()
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get combined tool definitions from all servers."""
        all_tools = []
        
        for server_name, server in self.servers.items():
            tools = server.list_tools()
            for tool in tools:
                tool_copy = tool.copy()
                # Store original name and server for later
                tool_copy["_mcp_server"] = server_name
                tool_copy["_mcp_original_name"] = tool.get("name", "unknown")
                # Prefix name to avoid collisions
                tool_copy["name"] = f"mcp_{server_name}_{tool.get('name', 'unknown')}"
                tool_copy["description"] = f"[{server_name}] {tool.get('description', 'MCP tool')}"
                all_tools.append(tool_copy)
        
        return all_tools
    
    def call_tool(self, full_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool by its prefixed name."""
        # Find the server that owns this tool
        for server_name, server in self.servers.items():
            prefix = f"mcp_{server_name}_"
            if full_name.startswith(prefix):
                tool_name = full_name[len(prefix):]
                return server.call_tool(tool_name, arguments)
        
        return {"error": f"Unknown MCP tool: {full_name}"}


# Global client instance
_mcp_client: Optional[MCPClient] = None


def init_mcp() -> Optional[MCPClient]:
    """Initialize MCP client and start servers."""
    global _mcp_client
    _mcp_client = MCPClient()
    _mcp_client.start_servers()
    return _mcp_client


def get_mcp_client() -> Optional[MCPClient]:
    """Get the global MCP client instance."""
    return _mcp_client


def stop_mcp():
    """Stop all MCP servers."""
    global _mcp_client
    if _mcp_client:
        _mcp_client.stop_servers()
        _mcp_client = None


def create_config_example():
    """Create example config file."""
    config_dir = Path.home() / ".config" / "coding-guy"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    example_path = config_dir / "config.json.example"
    if example_path.exists():
        return
    
    example = {
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
            }
        }
    }
    
    with open(example_path, "w") as f:
        json.dump(example, f, indent=2)
    
    print(f"[MCP] Created example config at: {example_path}", file=sys.stderr)
