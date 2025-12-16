import json, time, os

LOG_PATH = "/tmp/metrics.jsonl"

def log_metric(engine: str, latency_ms: float):
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine,
        "latency_ms": round(latency_ms, 1)
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logger] Failed: {e}")
