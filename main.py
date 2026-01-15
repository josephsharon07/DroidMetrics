from fastapi import FastAPI
from adb_utils import adb_shell
from parsers import parse_key_value_block, parse_cpu_freq, kb_to_mb

app = FastAPI(
    title="Android System API (ADB)",
    description="System information via ADB (no root, no app)",
    version="1.0.0"
)

# ---------------- DEVICE ----------------
@app.get("/device")
def device_info():
    return {
        "model": adb_shell("getprop ro.product.model"),
        "manufacturer": adb_shell("getprop ro.product.manufacturer"),
        "android_version": adb_shell("getprop ro.build.version.release"),
        "sdk": int(adb_shell("getprop ro.build.version.sdk")),
        "hardware": adb_shell("getprop ro.hardware"),
        "board": adb_shell("getprop ro.board.platform"),
    }


# ---------------- CPU ----------------
@app.get("/cpu")
def cpu_info():
    return {
        "cores": int(adb_shell("nproc")),
        "abi": adb_shell("getprop ro.product.cpu.abi"),
        "abi_list": adb_shell("getprop ro.product.cpu.abilist").split(","),
        "cpuinfo": adb_shell("cat /proc/cpuinfo")
    }


@app.get("/cpu/frequency")
def cpu_frequency():
    raw = adb_shell(
        "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; "
        "do echo $f: $(cat $f); done"
    )
    return parse_cpu_freq(raw)


# ---------------- MEMORY ----------------
@app.get("/memory")
def memory_info():
    meminfo = adb_shell("cat /proc/meminfo")
    data = {}

    for line in meminfo.splitlines():
        if line.startswith(("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")):
            key, val = line.split(":", 1)
            data[key] = kb_to_mb(val.strip().split()[0])

    return {
        "unit": "MB",
        "data": data
    }


# ---------------- STORAGE ----------------
@app.get("/storage")
def storage_info():
    out = adb_shell("df /data | tail -1").split()
    return {
        "filesystem": out[0],
        "total_kb": int(out[1]),
        "used_kb": int(out[2]),
        "free_kb": int(out[3])
    }


# ---------------- BATTERY ----------------
@app.get("/battery")
def battery_info():
    raw = adb_shell("dumpsys battery")
    return parse_key_value_block(raw)


# ---------------- THERMAL ----------------
@app.get("/thermal")
def thermal_info():
    return {
        "raw": adb_shell("dumpsys thermalservice")
    }


# ---------------- FULL SYSTEM ----------------
@app.get("/system")
def system_info():
    return {
        "device": device_info(),
        "cpu": cpu_info(),
        "cpu_frequency": cpu_frequency(),
        "memory": memory_info(),
        "storage": storage_info(),
        "battery": battery_info(),
        "thermal": thermal_info()
    }
