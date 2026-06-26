#!/usr/bin/env python3
"""Full-corpus vLLM causal-triple extractor. Resumable (skips note_ids already in
OUT), chunked, incremental flush. Reads .jsonl[.gz] of {note_id,text}, writes
{note_id, raw} (raw = model's JSON-array string) to OUT."""
import json, time, sys, gzip, os
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
IN = sys.argv[1] if len(sys.argv) > 1 else "full_input.jsonl.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "triples_llm.jsonl"
CHUNK = 8000

SYS = ("You are a clinical information extraction system. From the clinical note text, "
       "extract ONLY explicitly stated cause->effect medical relationships (cues such as "
       "'due to', 'secondary to', 'caused by', 'led to', 'resulting in', 'complicated by'). "
       "Output ONLY a JSON array of objects {\"cause\":\"...\",\"effect\":\"...\"} using the "
       "medical terms as written in the text. No commentary. If none, output [].")

def opener(p): return gzip.open(p, "rt") if p.endswith(".gz") else open(p)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["note_id"])
        except Exception: pass
    print(f"resume: {len(done)} notes already done", flush=True)

rows = []
with opener(IN) as f:
    for l in f:
        r = json.loads(l)
        if r["note_id"] not in done:
            rows.append(r)
print(f"to process: {len(rows)} notes", flush=True)

if not rows:
    print("nothing to do"); sys.exit(0)

llm = LLM(model=MODEL, quantization="awq_marlin", dtype="float16",
          gpu_memory_utilization=0.90, max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=512)

fout = open(OUT, "a")
t0 = time.time(); n_done = 0
for i in range(0, len(rows), CHUNK):
    chunk = rows[i:i+CHUNK]
    msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": r["text"]}] for r in chunk]
    outs = llm.chat(msgs, sp)
    for r, o in zip(chunk, outs):
        fout.write(json.dumps({"note_id": r["note_id"], "raw": o.outputs[0].text}) + "\n")
    fout.flush()
    n_done += len(chunk)
    el = time.time() - t0; rate = n_done / el
    eta = (len(rows) - n_done) / rate / 3600 if rate else 0
    print(f"[{n_done}/{len(rows)}] {rate:.2f} notes/s  ETA {eta:.2f} h", flush=True)
fout.close()
print("ALL DONE", flush=True)
