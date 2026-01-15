def parse_key_value_block(text: str):
    data = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def parse_cpu_freq(text: str):
    freqs = {}
    for line in text.splitlines():
        if ":" in line:
            path, value = line.split(":", 1)
            cpu = path.split("/")[-2]
            freqs[cpu] = int(value.strip())
    return freqs


def kb_to_mb(value: str):
    return round(int(value) / 1024, 2)
