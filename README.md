# Android System API (ADB)

Expose Android system information via HTTP using **ADB + FastAPI**.
No root. No Android app. No sensors.

---

## Requirements

- Python 3.9+
- ADB installed
- USB debugging enabled
- One Android device connected

Check:
```
adb devices
```

---

## Install

```
pip install -r requirements.txt
```

---

## Run

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### Device
```
GET /device
```

Returns model, Android version, hardware info.

---

### CPU
```
GET /cpu
GET /cpu/frequency
```

- Core count
- ABI
- Per-core real-time frequency (kHz)

---

### Memory
```
GET /memory
```

- Total RAM
- Available RAM
- Swap total/free

(Unit: MB)

---

### Storage
```
GET /storage
```

- Internal storage (/data)

---

### Battery
```
GET /battery
```

- Level
- Status
- Voltage
- Temperature
- Health

---

### Thermal
```
GET /thermal
```

- CPU / Battery / Skin temperatures
- Throttling status (if present)

---

### Full System
```
GET /system
```

Returns everything above in one JSON response.

---

## What This API CAN Do

- CPU frequency (per-core)
- RAM & Swap
- Storage
- Battery
- Thermal sensors

---

## What This API CANNOT Do

- CPU usage %
- Live sensor values
- GPU usage / frequency

These require:
- Root OR
- Privileged Android app

---

## Swagger UI

http://localhost:8000/docs
