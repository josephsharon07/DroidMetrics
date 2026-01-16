# WebSocket Connection Test & Fix

## Issue
WebSocket endpoints (`/ws/metrics` and `/ws/cpu`) returning HTTP 404 when accessed.

## Root Cause
The API server needs to be restarted to load the updated main.py with the WebSocket endpoints properly registered.

## Solution Applied

### 1. Added CORS Support
- Added `CORSMiddleware` to main.py (handles both HTTP and WebSocket CORS)
- Configuration allows all origins, methods, and headers

### 2. Verified WebSocket Endpoints
- `/ws/metrics` - Real-time metrics (battery, memory, storage, CPU, thermal) - 2 second updates
- `/ws/cpu` - Real-time CPU frequencies - 1 second updates

### 3. Updated Dashboard
- Added better WebSocket connection handling
- Implements automatic reconnection with exponential backoff
- Falls back to HTTP polling if WebSocket fails
- Shows connection status for both endpoints

## How to Test

### Step 1: Restart the API Server
```bash
# Kill existing server
pkill -f "uvicorn main:app"

# Start fresh
cd /home/sharon/Documents/ADB-Insight
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 2: Test with Python
```bash
python3 << 'PYEOF'
import asyncio
import json
import websockets

async def test():
    try:
        # Test /ws/metrics
        async with websockets.connect("ws://10.10.10.48:8000/ws/metrics") as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            print("✅ /ws/metrics connected")
            print(json.dumps(json.loads(msg), indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

asyncio.run(test())
PYEOF
```

### Step 3: Test with Dashboard
Open `dashboard.html` in your browser:
```bash
# Option 1: Direct file
file:///home/sharon/Documents/ADB-Insight/dashboard.html

# Option 2: HTTP server
cd /home/sharon/Documents/ADB-Insight
python3 -m http.server 8080
# Then visit: http://localhost:8080/dashboard.html
```

Check browser console (F12) for WebSocket connection logs.

## Expected Behavior

### When Working
- Dashboard loads and connects to API
- Battery, memory, storage percentages update every 2 seconds (WebSocket /ws/metrics)
- CPU frequencies update every 1 second (WebSocket /ws/cpu)
- Status indicators show ✅ green for both WebSocket connections

### When WebSocket Fails (Graceful Fallback)
- Dashboard still works with HTTP polling
- Status indicators show ❌ red for WebSocket
- Data still updates every 5 seconds via HTTP /cpu/frequency, /battery, /memory, /storage endpoints
- Warning message appears: "⚠️ WebSocket disconnected - using HTTP polling"

## Key Files Updated
- `main.py`: Added CORS middleware, verified WebSocket endpoints
- `dashboard.html`: Enhanced WebSocket error handling and fallback logic

## WebSocket Data Format

### /ws/metrics (2s updates)
```json
{
  "timestamp": "2026-01-16T10:30:45.123456",
  "battery_level": 51,
  "memory_usage_percent": 61.09,
  "storage_usage_percent": 30.46,
  "cpu_avg_mhz": 546.0,
  "cpu_max_mhz": 546.0,
  "cpu_min_mhz": 546.0,
  "thermal_max_temp": 44.1
}
```

### /ws/cpu (1s updates)
```json
{
  "timestamp": "2026-01-16T10:30:46.123456",
  "per_core": {
    "cpu": 546000
  },
  "min_mhz": 546.0,
  "max_mhz": 546.0,
  "avg_mhz": 546.0,
  "core_count": 1
}
```

