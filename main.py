from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

from adb_utils import adb_shell, adb_devices
from parsers import (
    parse_key_value_block, parse_cpu_freq, parse_cpu_frequencies_detailed, parse_thermal_data, 
    parse_battery_level, kb_to_mb, kb_to_gb
)

logger = logging.getLogger(__name__)

# ============ PYDANTIC MODELS ============
class DeviceInfo(BaseModel):
    model: str = Field(..., example="SM-F127G")
    manufacturer: str = Field(..., example="samsung")
    android_version: str = Field(..., example="13")
    sdk: int = Field(..., example=33)
    hardware: str = Field(..., example="exynos850")
    board: str = Field(..., example="universal3830")

    class Config:
        json_schema_extra = {"example": {
            "model": "SM-F127G",
            "manufacturer": "samsung",
            "android_version": "13",
            "sdk": 33,
            "hardware": "exynos850",
            "board": "universal3830"
        }}


class CPUInfo(BaseModel):
    cores: int = Field(..., example=8)
    abi: str = Field(..., example="arm64-v8a")
    abi_list: List[str] = Field(..., example=["arm64-v8a", "armeabi-v7a"])
    arch: str = Field(..., example="ARMv8")


class CPUFrequency(BaseModel):
    per_core: Dict[str, int] = Field(..., example={"cpu0": 1800000, "cpu1": 1800000})
    min_khz: int = Field(..., example=546000)
    max_khz: int = Field(..., example=1800000)
    min_mhz: float = Field(..., example=546.0)
    max_mhz: float = Field(..., example=1800.0)
    avg_mhz: float = Field(..., example=1173.25)
    core_count: int = Field(..., example=8)


class MemoryInfo(BaseModel):
    total_mb: float = Field(..., example=3704.71)
    available_mb: float = Field(..., example=1236.54)
    used_mb: float = Field(..., example=2468.17)
    usage_percent: float = Field(..., example=66.6)
    swap_total_mb: float = Field(..., example=4096.0)
    swap_free_mb: float = Field(..., example=2729.53)


class StorageInfo(BaseModel):
    filesystem: str = Field(..., example="/dev/block/dm-44")
    total_gb: float = Field(..., example=51.35)
    used_gb: float = Field(..., example=15.63)
    free_gb: float = Field(..., example=35.52)
    usage_percent: float = Field(..., example=30.5)


class BatteryInfo(BaseModel):
    level: int = Field(..., example=39, ge=0, le=100)
    health: str = Field(..., example="Good")
    status: str = Field(..., example="Charging")
    voltage_mv: int = Field(..., example=3960)
    temperature_c: float = Field(..., example=34.6)
    technology: str = Field(..., example="Li-ion")
    is_charging: bool = Field(..., example=True)


class ThermalInfo(BaseModel):
    temperatures: Dict[str, float] = Field(..., example={"AP": 39.5, "BAT": 34.6, "SKIN": 35.4})
    max_temp_c: float = Field(..., example=39.5)
    min_temp_c: float = Field(..., example=34.6)


class HealthStatus(BaseModel):
    status: str = Field(..., example="healthy")
    adb_connected: bool = Field(..., example=True)
    timestamp: datetime = Field(...)


class RealTimeMetrics(BaseModel):
    """Real-time system metrics for streaming."""
    timestamp: datetime
    battery_level: int
    memory_usage_percent: float
    storage_usage_percent: float
    cpu_avg_mhz: float
    cpu_max_mhz: float
    cpu_min_mhz: float
    thermal_max_temp: float


class SystemInfo(BaseModel):
    device: DeviceInfo
    cpu: CPUInfo
    cpu_frequency: CPUFrequency
    memory: MemoryInfo
    storage: StorageInfo
    battery: BatteryInfo
    thermal: ThermalInfo
    timestamp: datetime = Field(...)


# ============ FastAPI APP ============
app = FastAPI(
    title="Android System API (ADB)",
    description="Real-time Android system metrics via ADB (no root, no app required)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============ CORS MIDDLEWARE ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
)


# ============ ERROR HANDLERS ============
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error": "ADB Error"}
    )


# ============ HEALTH & DIAGNOSTICS ============
@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """Check API health and ADB connection status."""
    try:
        devices = adb_devices()
        is_connected = devices and "device" in devices
        status_str = "healthy" if is_connected else "degraded"
        
        return HealthStatus(
            status=status_str,
            adb_connected=is_connected,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADB connection failed"
        )


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with available endpoints."""
    return {
        "app": "Android System API (ADB)",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "device": "/device",
            "cpu": "/cpu",
            "memory": "/memory",
            "storage": "/storage",
            "battery": "/battery",
            "thermal": "/thermal",
            "system": "/system"
        },
        "docs": "/docs",
        "timestamp": datetime.now()
    }


# ============ DEVICE ENDPOINT ============
@app.get("/device", response_model=DeviceInfo, tags=["Device"])
async def device_info():
    """
    Get device information (model, manufacturer, Android version, etc).
    
    No root required. Uses system properties.
    """
    try:
        return DeviceInfo(
            model=adb_shell("getprop ro.product.model"),
            manufacturer=adb_shell("getprop ro.product.manufacturer"),
            android_version=adb_shell("getprop ro.build.version.release"),
            sdk=int(adb_shell("getprop ro.build.version.sdk")),
            hardware=adb_shell("getprop ro.hardware"),
            board=adb_shell("getprop ro.board.platform"),
        )
    except Exception as e:
        logger.error(f"Device info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CPU ENDPOINTS ============
@app.get("/cpu", response_model=CPUInfo, tags=["CPU"])
async def cpu_info():
    """
    Get CPU information (cores, architecture, ABI).
    
    Returns: core count, ABI, and supported architectures.
    """
    try:
        cores = int(adb_shell("nproc"))
        abi = adb_shell("getprop ro.product.cpu.abi")
        abi_list = [a.strip() for a in adb_shell("getprop ro.product.cpu.abilist").split(",")]
        
        arch_map = {
            "arm64-v8a": "ARMv8",
            "armeabi-v7a": "ARMv7",
            "x86_64": "x86-64",
            "x86": "x86"
        }
        arch = arch_map.get(abi, "Unknown")
        
        return CPUInfo(
            cores=cores,
            abi=abi,
            abi_list=abi_list,
            arch=arch
        )
    except Exception as e:
        logger.error(f"CPU info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cpu/frequency", response_model=CPUFrequency, tags=["CPU"])
async def cpu_frequency():
    """
    Get real-time CPU frequency for each core with statistics.
    
    Returns: per-core frequencies in kHz, plus min, max, and average calculations.
    """
    try:
        raw = adb_shell(
            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; "
            "do echo $f: $(cat $f); done"
        )
        freq_data = parse_cpu_frequencies_detailed(raw)
        
        if "error" in freq_data:
            raise HTTPException(status_code=500, detail="Failed to parse CPU frequencies")
        
        return CPUFrequency(
            per_core=freq_data["per_core"],
            min_khz=freq_data["min_khz"],
            max_khz=freq_data["max_khz"],
            min_mhz=freq_data["min_mhz"],
            max_mhz=freq_data["max_mhz"],
            avg_mhz=freq_data["avg_mhz"],
            core_count=freq_data["core_count"]
        )
    except Exception as e:
        logger.error(f"CPU frequency error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ MEMORY ENDPOINT ============
@app.get("/memory", response_model=MemoryInfo, tags=["Memory"])
async def memory_info():
    """
    Get RAM and swap memory information.
    
    Returns: total, available, used memory in MB, plus swap info.
    """
    try:
        meminfo = adb_shell("cat /proc/meminfo")
        data = {}
        
        for line in meminfo.splitlines():
            if line.startswith(("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")):
                key, val = line.split(":", 1)
                data[key] = kb_to_mb(val.strip().split()[0])
        
        total = data.get("MemTotal", 0)
        available = data.get("MemAvailable", 0)
        used = total - available
        usage_percent = round((used / total * 100), 2) if total > 0 else 0
        
        return MemoryInfo(
            total_mb=total,
            available_mb=available,
            used_mb=used,
            usage_percent=usage_percent,
            swap_total_mb=data.get("SwapTotal", 0),
            swap_free_mb=data.get("SwapFree", 0)
        )
    except Exception as e:
        logger.error(f"Memory info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ STORAGE ENDPOINT ============
@app.get("/storage", response_model=StorageInfo, tags=["Storage"])
async def storage_info():
    """
    Get internal storage (/data) information.
    
    Returns: total, used, free space in GB with usage percentage.
    """
    try:
        out = adb_shell("df /data | tail -1").split()
        total_kb = int(out[1])
        used_kb = int(out[2])
        free_kb = int(out[3])
        
        usage_percent = round((used_kb / total_kb * 100), 2) if total_kb > 0 else 0
        
        return StorageInfo(
            filesystem=out[0],
            total_gb=kb_to_gb(total_kb),
            used_gb=kb_to_gb(used_kb),
            free_gb=kb_to_gb(free_kb),
            usage_percent=usage_percent
        )
    except Exception as e:
        logger.error(f"Storage info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ BATTERY ENDPOINT ============
@app.get("/battery", response_model=BatteryInfo, tags=["Battery"])
async def battery_info():
    """
    Get battery status and health.
    
    Returns: charge level, health, status, voltage, temperature, and charging state.
    """
    try:
        raw = adb_shell("dumpsys battery")
        battery_data = parse_key_value_block(raw)
        battery = parse_battery_level(battery_data)
        
        return BatteryInfo(
            level=battery["level"],
            health=battery["health"],
            status=battery["status"],
            voltage_mv=battery["voltage_mv"],
            temperature_c=battery["temperature_c"],
            technology=battery["technology"],
            is_charging=battery["is_charging"]
        )
    except Exception as e:
        logger.error(f"Battery info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ THERMAL ENDPOINT ============
@app.get("/thermal", response_model=ThermalInfo, tags=["Thermal"])
async def thermal_info():
    """
    Get thermal sensor readings.
    
    Returns: CPU, battery, and skin temperatures with min/max values.
    """
    try:
        raw = adb_shell("dumpsys thermalservice")
        temps = parse_thermal_data(raw)
        
        if not temps or "raw" in temps:
            raise HTTPException(status_code=500, detail="Failed to parse thermal data")
        
        temp_values = [t["value"] for t in temps.values()]
        max_temp = max(temp_values) if temp_values else 0
        min_temp = min(temp_values) if temp_values else 0
        
        simple_temps = {name: data["value"] for name, data in temps.items()}
        
        return ThermalInfo(
            temperatures=simple_temps,
            max_temp_c=max_temp,
            min_temp_c=min_temp
        )
    except Exception as e:
        logger.error(f"Thermal info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WEBSOCKET REAL-TIME STREAMING ============
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time system metrics streaming.
    
    Sends battery, memory, storage, CPU, and thermal data every 2 seconds.
    Connect: ws://localhost:8000/ws/metrics
    """
    await websocket.accept()
    logger.info("WebSocket client connected for metrics")
    
    try:
        while True:
            try:
                # Get current metrics
                memory = await memory_info()
                storage = await storage_info()
                battery = await battery_info()
                cpu_freq = await cpu_frequency()
                thermal = await thermal_info()
                
                # Create real-time metrics object
                metrics = RealTimeMetrics(
                    timestamp=datetime.now(),
                    battery_level=battery.level,
                    memory_usage_percent=memory.usage_percent,
                    storage_usage_percent=storage.usage_percent,
                    cpu_avg_mhz=cpu_freq.avg_mhz,
                    cpu_max_mhz=cpu_freq.max_mhz,
                    cpu_min_mhz=cpu_freq.min_mhz,
                    thermal_max_temp=thermal.max_temp_c
                )
                
                # Send metrics as JSON
                await websocket.send_json(metrics.model_dump(mode='json'))
                
                # Wait 2 seconds before next update
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await websocket.send_json({"error": str(e)})
                await asyncio.sleep(2)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1000)
        except:
            pass


@app.websocket("/ws/cpu")
async def websocket_cpu(websocket: WebSocket):
    """
    WebSocket endpoint for real-time CPU frequency streaming.
    
    Sends per-core CPU frequencies, min, max, and average every 1 second.
    Connect: ws://localhost:8000/ws/cpu
    """
    await websocket.accept()
    logger.info("WebSocket client connected for CPU metrics")
    
    try:
        while True:
            try:
                cpu_freq = await cpu_frequency()
                
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "per_core": cpu_freq.per_core,
                    "min_mhz": cpu_freq.min_mhz,
                    "max_mhz": cpu_freq.max_mhz,
                    "avg_mhz": cpu_freq.avg_mhz,
                    "core_count": cpu_freq.core_count
                }
                
                await websocket.send_json(payload)
                
                # Update every 1 second for CPU
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting CPU metrics: {e}")
                await websocket.send_json({"error": str(e)})
                await asyncio.sleep(1)
                
    except WebSocketDisconnect:
        logger.info("WebSocket CPU client disconnected")
    except Exception as e:
        logger.error(f"WebSocket CPU error: {e}")
        try:
            await websocket.close(code=1000)
        except:
            pass


# ============ FULL SYSTEM ENDPOINT ============
@app.get("/system", response_model=SystemInfo, tags=["System"])
async def system_info():
    """
    Get complete system information.
    
    Returns: All device metrics (device, CPU, memory, storage, battery, thermal).
    Single endpoint for comprehensive system overview.
    """
    try:
        device = await device_info()
        cpu = await cpu_info()
        cpu_freq = await cpu_frequency()
        memory = await memory_info()
        storage = await storage_info()
        battery = await battery_info()
        thermal = await thermal_info()
        
        return SystemInfo(
            device=device,
            cpu=cpu,
            cpu_frequency=cpu_freq,
            memory=memory,
            storage=storage,
            battery=battery,
            thermal=thermal,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
