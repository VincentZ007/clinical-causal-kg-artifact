#!/usr/bin/env python3
"""Batched generation via transformers (no vllm). Loads Qwen2.5-7B-Instruct-AWQ.
Reads prompts.jsonl ({qid,system,type,messages}) -> answers.jsonl ({qid,system,type,answer}).
Usage: python gen_hpc_hf.py prompts.jsonl answers.jsonl [batch_size]"""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
inp = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
outp = sys.argv[2] if len(sys.argv) > 2 else "answers.jsonl"
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 48

rows = [json.loads(l) for l in open(inp)]
print(f"{len(rows)} prompts  batch={BS}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model.eval()
print("model loaded", flush=True)

prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
out = []
t0 = time.time()
for i in range(0, len(rows), BS):
    chunk = prompts[i:i + BS]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[1]
    for j, g in enumerate(gen):
        text = tok.decode(g[plen:], skip_special_tokens=True).strip().split("\n")[0]
        r = rows[i + j]
        out.append({"qid": r["qid"], "system": r["system"], "type": r["type"], "answer": text})
    if i % (BS * 20) == 0:
        print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
with open(outp, "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"wrote {outp}  ({time.time()-t0:.0f}s total)", flush=True)
