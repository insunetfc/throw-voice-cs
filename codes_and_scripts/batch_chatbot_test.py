import requests, csv, time, statistics, os, sys, json

# Set your tunnel URL here:
URL = "https://honest-trivially-buffalo.ngrok-free.app/respond"
INPUT = os.environ.get("CSV", "/home/tiongsik/Python/outbound_calls/chatbot/data/intent_dataset.csv")   # expects headers like: question,label
OUTPUT = os.environ.get("OUTPUT", "./responses_log.txt")
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "0"))

headers = {"Content-Type": "application/json"}

def ask(text):
    try:
        r = requests.post(URL, headers=headers, json={"text": text}, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}", "text": text}
    except Exception as e:
        return {"error": str(e), "text": text}

def main():
    if not os.path.exists(INPUT):
        print(f"[error] Missing input file: {INPUT}")
        return

    lines = []
    t0 = time.time()
    with open(INPUT, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not ("question" in reader.fieldnames and "label" in reader.fieldnames):
            print(f"[warn] No headers detected. Using first two columns as question,label.")
            f.seek(0)
            reader = csv.reader(f)
            data = [(r[0], r[1]) for r in reader]
        else:
            data = [(r["question"], r["label"]) for r in reader]

    if SAMPLE_LIMIT:
        data = data[:SAMPLE_LIMIT]

    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write(f"# Evaluation started {time.ctime()}\n")
        out.write(f"# Endpoint: {URL}\n")
        out.write(f"# Samples: {len(data)}\n\n")

        for i, (text, gold) in enumerate(data, 1):
            print(f"[{i}/{len(data)}] {text[:50]} ...")
            t1 = time.time()
            resp = ask(text)
            dt = time.time() - t1
            resp_str = json.dumps(resp, ensure_ascii=False, indent=2)
            out.write(f"[{i}] question: {text}\n")
            out.write(f"    gold_label: {gold}\n")
            out.write(f"    latency: {dt:.3f}s\n")
            out.write(resp_str + "\n\n")

    print(f"✅ Done. Saved to {OUTPUT} ({len(data)} samples, {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
