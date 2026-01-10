# 2) Your post-Phase2 “do this next” plan (Phase 2.5 then Phase 3)

You are currently “lost” because you’re mixing three goals:
- build reasoning,
- extend context,
- align behavior.

Here is the clean sequence that matches the Gemini plan **and** won’t make your model brittle.

## Phase 2 (you’re finishing this)
Goal: reasoning core at 2k.

**When Phase 2 completes:**
1) Save a final checkpoint tag, e.g. `phase2_final.pt`.

---

## Phase 2.5 (Bridge): 8k context + tool/spec protocol exposure (≈ 1B tokens)

### Goal
Teach:
- **8k position robustness**
- **tool/protocol grammar**
- keep reasoning/code alive via replay

### Dataset to train on (build this)
Create `data/shards/phase2_5_mixed/{train,val}` with approx weights:
- 50% Phase2 reasoning replay
- 25% tool/protocol (toolprep or glaive-style)
- 25% code / longctx hard-deps

### Exactly what to do

#### Step A — Build LongCtx Hard-Dependency shards (8k-capable)
Run the longctx-hard script we made earlier (or keep its output as “phase2.5_longctx_hard”).

Example:
```bash
python3 scripts/data_fetch/phase3_longctx_hard_shards.py \
  --out_dir data/shards/phase2_5_longctx_hard \
  --target_train_tokens 400000000 \
  --target_val_tokens 20000000
```

(Yes, the filename says phase3; the data is usable for phase2.5. If you want I can rename it, but it’s not required.)

#### Step B — Ensure tool/protocol shards exist (toolprep)
If you already created `data/shards/toolprep/{train,val}`, you’re good.

If not, run your `tool_synth_shards.py` to produce it.

#### Step C — Ensure you have a code-heavy stream
If you already have stack-edu code in Phase2 shards: fine. If not, generate a code-only token stream (stack-edu) into `data/shards/phase2_code`.

#### Step D — Interleave into Phase 2.5 mixed
Use the fixed interleaver:

```bash
python3 scripts/data_manipulation/interleave_bins_v2.py \
  --out_dir data/shards/phase2_5_mixed \
  --target_train_tokens 1000000000 \
  --target_val_tokens 50000000 \
  --source phase2=data/shards/phase2_v2:0.50 \
  --source toolprep=data/shards/toolprep:0.25 \
  --source longctx=data/shards/phase2_5_longctx_hard:0.25
```

#### Step E — Train Phase 2.5 at seq_len=8192
In `params.py`, set `MODE = "longctx_5090"` (ignore the name; it’s your 8k config).

Then run training **model-only init**:

```bash
python3 train.py \
  --train_dir data/shards/phase2_5_mixed/train \
  --val_dir data/shards/phase2_5_mixed/val \
  --init_from models/phase2_final.pt \
  --total_opt_steps 2000 \
  --val_every_opt 200
```

Notes:
- Use `--init_from` so optimizer/scheduler reset (bridge needs different LR dynamics).
- If 8k OOMs on 16GB, reduce compile, reduce workers, keep batch=1.

---

## Phase 3 (Refine): LongCtx reasoning + Generic SFT (≈ 0.5B tokens)

### Key change from earlier:
**Phase 3 SFT should be mostly generic.**  
Your synthetic planning-spec should be treated primarily as:
- **Phase 2.5 protocol exposure**, not “the SFT personality”.

### Dataset build
You want:
- **LongCtx Reasoning**: 300–400M tokens (harddeps + arxiv + code long)
- **Generic SFT**: 50–100M tokens (OpenHermes/UltraChat style)
- Optional: 0–10M tokens of your planning spec if you still want it to be “snappy” on that format.

### What I recommend (given your stated goal)
- Put your synthetic planning spec **mostly in Phase 2.5**.
- In Phase 3 SFT, use **generic instruction chat** + **some tool calling**.
- Only include planning spec in Phase 3 if you see the model “forgets” the exact delimiter format.

### Exactly what to do

1) Build `data/shards/phase3_longctx_hard/{train,val}` (same script as above, lower tokens).
2) Build `data/shards/phase3_sft/{train,val}` using **generic SFT datasets** (not your synthetic).

Right now the `phase3_sft_shards.py` I gave earlier is **synthetic-first**. That’s not aligned with what you want. I will rewrite it so:
- synthetic is **off by default**
- OpenHermes (or similar) is the default base
- glaive optional

If you want, I’ll produce that updated file next message (so it’s copy-paste ready).

3) Interleave Phase3 final:
```bash
python3 scripts/data_manipulation/interleave_bins_v2.py \
  --out_dir data/shards/phase3_mixed \
  --target_train_tokens 500000000 \
  --target_val_tokens 25000000 \
  --source phase1=data/shards/phase1:0.10 \
  --source phase2=data/shards/phase2_v2:0.15 \
  --source longctx=data/shards/phase3_longctx_hard:0.50 \
  --source sft=data/shards/phase3_sft:0.25
```

4) Train Phase 3 using `--init_from` from Phase2.5 best checkpoint:
```bash
python3 train.py \
  --train_dir data/shards/phase3_mixed/train \
  --val_dir data/shards/phase3_mixed/val \
  --init_from models/best_step_XXXX.pt \
  --total_opt_steps 1000 \
  --val_every_opt 100
```

---

# 3) What will the model “feel like” to chat with?

If you do the plan above:

- After **Phase 2**: smart-ish continuation model, not reliably assistant-like.
- After **Phase 2.5**: it should handle **8k context** and understand tool/spec “grammar”, but may still be blunt/continuation-y.
- After **Phase 3**: it should feel like a normal assistant for normal prompts, and **snap into structured formats** when you explicitly ask (because it learned that behavior from generic SFT + protocol exposure).

If instead you make Phase 3 SFT heavily synthetic-spec, it will “feel” like a formatter, not an assistant. That’s what we’re avoiding.

**Pure generic SFT** (recommended for “assistant feel”):  
   - 90% OpenHermes/UltraChat-ish instruction chat  
   - 10% Glaive tool-calling  
   - 0% synthetic planning spec