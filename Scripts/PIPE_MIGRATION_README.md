# Tempest AI - Socket to Pipe Migration

## Overview

The Tempest AI system has been refactored to use named pipes instead of sockets for communication between the MAME Lua script and the Python AI server. This change resolves the socket connection issues that were causing "No such file or directory" errors.

## Changes Made

### 1. New Pipe Server (`pipe_server.py`)
- Replaces the socket-based communication with named pipes
- Creates temporary pipes in the system temp directory
- Handles cross-platform compatibility (Unix FIFOs vs Windows files)
- Maintains the same communication protocol as the original socket server

### 2. Updated Main Application (`main.py`)
- Now imports and uses `PipeServer` instead of `SocketServer`
- All other functionality remains the same

### 3. New Lua Script (`main_pipes.lua`)
- Pipe-based version of the original `main.lua`
- Uses MAME's `emu.file()` API to open pipes for communication
- Automatically detects pipe paths from system temp directory
- Maintains all game logic and state management from the original

## File Structure

```
Scripts/
├── main.py                    # Updated main application (uses pipes)
├── pipe_server.py            # New pipe-based server
├── main_pipes.lua            # New pipe-based Lua script
├── main.lua                  # Original socket-based Lua script (kept for reference)
├── socket_server.py          # Original socket server (kept for reference)
└── PIPE_MIGRATION_README.md  # This file
```

## How to Use

### 1. Start the Python AI Server
```bash
cd Scripts
python main.py
```

The server will:
- Create named pipes in the system temp directory
- Wait for MAME to connect
- Display "Waiting for MAME to connect..." message

### 2. Start MAME with the New Lua Script
```bash
mame tempest1 -autoboot_script Scripts/main_pipes.lua -skip_gameinfo
```

## Communication Flow

1. **Python Server Startup**:
   - Creates named pipes: `tempest_ai_input` and `tempest_ai_output` in temp directory
   - Opens pipes for reading/writing
   - Waits for MAME connection

2. **MAME Lua Script Startup**:
   - Detects pipe paths from system temp directory
   - Opens pipes for communication
   - Begins game state monitoring

3. **Runtime Communication**:
   - Lua script sends game state data to Python via output pipe
   - Python processes data and sends actions back via input pipe
   - Same binary protocol as original socket implementation

## Pipe Paths

The system automatically uses the following pipe paths:
- **Unix/Linux/macOS**: `/tmp/tempest_ai_input` and `/tmp/tempest_ai_output`
- **Windows**: `%TEMP%\tempest_ai_input` and `%TEMP%\tempest_ai_output`

## Advantages of Pipe-Based Communication

1. **Reliability**: Named pipes are more reliable than sockets for local IPC
2. **Simplicity**: No network configuration or port management required
3. **Performance**: Direct kernel-level communication without network stack overhead
4. **Security**: Local-only communication, no network exposure

## Troubleshooting

### Common Issues

1. **"Failed to open input/output pipe"**
   - Ensure Python server is running first
   - Check that temp directory is writable
   - Verify no other processes are using the pipe files

2. **"Pipe read timeout"**
   - Python server may have crashed or stopped
   - Restart the Python server
   - Check Python console for error messages

3. **"MAME interface not available"**
   - Standard MAME Lua script issue
   - Ensure correct MAME version and ROM
   - Check MAME console for Lua errors

### Debug Mode

To enable debug output in the Lua script, uncomment the debug print statements:
```lua
-- Uncomment these lines for debugging:
-- print("Pipes closed.")
-- print(string.format("[DEBUG] ..."))
```

## Backward Compatibility

The original socket-based files are preserved:
- `main.lua` - Original socket-based Lua script
- `socket_server.py` - Original socket server

To revert to socket-based communication:
1. Update `main.py` to import `SocketServer` instead of `PipeServer`
2. Use `main.lua` instead of `main_pipes.lua` with MAME

## Performance Notes

- Pipe communication should have lower latency than sockets
- No network buffer management required
- Direct file system I/O for communication
- Same binary protocol ensures no data format changes

## Future Improvements

Potential enhancements for the pipe-based system:
1. Automatic pipe cleanup on system restart
2. Multiple MAME instance support with unique pipe names
3. Pipe permission management for multi-user systems
4. Fallback to socket communication if pipes fail
