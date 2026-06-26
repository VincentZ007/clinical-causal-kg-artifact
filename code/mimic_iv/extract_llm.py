#!/usr/bin/env python3
"""vLLM causal-triple extractor — pilot/benchmark. Reads pilot_input.jsonl,
extracts cause->effect triples, separately times model-load vs inference,
and extrapolates inference time to the full corpus."""
import json, time, sys
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
IN = sys.argv[1] if len(sys.argv) > 1 else "pilot_input.jsonl"
OUT = "pilot_triples.jsonl"
TOTAL = 331793   # full discharge-note corpus

SYS = ("You are a clinical information extraction system. From the clinical note text, "
       "extract ONLY explicitly stated cause->effect medical relationships (cues such as "
       "'due to', 'secondary to', 'caused by', 'led to', 'resulting in', 'complicated by'). "
       "Output ONLY a JSON array of objects {\"cause\":\"...\",\"effect\":\"...\"} using the "
       "medical terms as written in the text. No commentary. If none, output [].")

rows = [json.loads(l) for l in open(IN)]
print(f"loaded {len(rows)} notes", flush=True)

t0 = time.time()
llm = LLM(model=MODEL, quantization="awq_marlin", dtype="float16",
          gpu_memory_utilization=0.90, max_model_len=4096, enforce_eager=False)
load_t = time.time() - t0

sp = SamplingParams(temperature=0.0, max_tokens=512)
msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": r["text"]}] for r in rows]

t1 = time.time()
outs = llm.chat(msgs, sp)
inf_t = time.time() - t1

in_tok = sum(len(o.prompt_token_ids) for o in outs)
out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
n = len(rows)
with open(OUT, "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"note_id": r["note_id"], "raw": o.outputs[0].text}) + "\n")

rate = n / inf_t
print("\n================ PILOT BENCHMARK ================")
print(f"model load        : {load_t:.1f} s (one-time)")
print(f"notes processed   : {n}")
print(f"inference time    : {inf_t:.1f} s")
print(f"throughput        : {rate:.2f} notes/s")
print(f"tokens: in={in_tok} out={out_tok} | {(in_tok+out_tok)/inf_t:.0f} tok/s total, {out_tok/inf_t:.0f} tok/s out")
print(f"avg out tokens/note: {out_tok/n:.0f}")
print(f"\n--- extrapolate to FULL corpus ({TOTAL} notes) on THIS 1 MIG slice ---")
print(f"estimated inference: {TOTAL/rate/3600:.2f} h  (+ {load_t:.0f}s one-time load)")
print(f"sample extractions:")
for o in outs[:3]:
    print("  ", o.outputs[0].text.replace(chr(10), " ")[:160])
