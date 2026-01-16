def parse_key_value_block(text: str):
    """Parse key-value pairs from dumpsys output."""
    data = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            # Skip empty keys and values
            if key and val:
                data[key] = val
    return data


def parse_cpu_freq(text: str):
    """Parse CPU frequency from scaling_cur_freq files."""
    freqs = {}
    for line in text.splitlines():
        if ":" in line:
            path, value = line.split(":", 1)
            try:
                # Extract cpu number from path like /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
                # Split path and find "cpu0", "cpu1", etc.
                path_parts = path.strip().split("/")
                cpu_part = None
                for part in path_parts:
                    if part.startswith("cpu") and part[3:].isdigit():
                        cpu_part = part
                        break
                
                if cpu_part:
                    freqs[cpu_part] = int(value.strip())
            except (ValueError, IndexError):
                continue
    return freqs if freqs else {"error": "Could not parse CPU frequencies"}


def parse_cpu_frequencies_detailed(text: str) -> dict:
    """Parse CPU frequencies with min/max calculations."""
    freqs = parse_cpu_freq(text)
    
    if "error" in freqs:
        return freqs
    
    # Extract numeric frequencies only
    freq_values = [f for f in freqs.values() if isinstance(f, int)]
    
    if not freq_values:
        return {"error": "No valid frequencies found"}
    
    return {
        "per_core": freqs,
        "min_khz": min(freq_values),
        "max_khz": max(freq_values),
        "min_mhz": round(min(freq_values) / 1000, 2),
        "max_mhz": round(max(freq_values) / 1000, 2),
        "avg_mhz": round(sum(freq_values) / len(freq_values) / 1000, 2),
        "core_count": len(freq_values)
    }


def parse_thermal_data(text: str):
    """Parse thermal service output and extract temperatures."""
    temps = {}
    for line in text.splitlines():
        if "Temperature{" in line:
            # Extract temperature data from format: Temperature{mValue=39.5, mType=0, mName=AP, mStatus=0}
            try:
                parts = line.split("Temperature{")[1].rstrip("}")
                items = [item.strip() for item in parts.split(", ")]
                temp_data = {}
                for item in items:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        temp_data[k.strip()] = v.strip()
                
                if "mName" in temp_data:
                    name = temp_data["mName"]
                    temps[name] = {
                        "value": float(temp_data.get("mValue", 0)),
                        "type": temp_data.get("mType", ""),
                        "status": temp_data.get("mStatus", "")
                    }
            except (IndexError, ValueError):
                continue
    return temps if temps else {"raw": text[:500]}


def parse_battery_level(data: dict) -> dict:
    """Extract key battery metrics from parsed data."""
    return {
        "level": int(data.get("level", 0)),
        "health": data.get("health", "unknown"),
        "status": data.get("status", "unknown"),
        "voltage_mv": int(data.get("voltage", 0)),
        "temperature_c": round(int(data.get("temperature", 0)) / 10, 1),
        "technology": data.get("technology", "unknown"),
        "is_charging": data.get("AC powered", "").lower() == "true" or 
                       data.get("USB powered", "").lower() == "true"
    }


def kb_to_mb(value: str) -> float:
    """Convert KB to MB."""
    try:
        return round(int(value) / 1024, 2)
    except ValueError:
        return 0.0


def kb_to_gb(value: int) -> float:
    """Convert KB to GB."""
    return round(value / (1024 * 1024), 2)
