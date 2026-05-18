# data_mix — Anti-overtraining SFT corpus for Gemma 4 E2B VLM

Engineering notes for the mixed-source SFT corpus. Companion to the
code in `src/data_mix/`. The goal: break PlantNet's monopoly on the
LoRA subspace so the fine-tuned Gemma 4 E2B doesn't answer "plant"
to every prompt.

This is the **data side** of the anti-forgetting stack. The finetune-
side companion (KL output-distribution penalty + L2 weight anchor +
camera-state prefix wiring) is in
[`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md).

## Files

### Mechanism / design

| File | Covers |
|---|---|
| [`01-data-prefix.md`](01-data-prefix.md) | **v4 camera-state prefix gate** (`[camera=on]` / `[camera=off]`, dispatched on image presence). Full plumbing trace from config → JSONL record → chat-template tokens. Required reading for anyone touching the iOS side. |
| [`02-bucket-design.md`](02-bucket-design.md) | Per-bucket implementation contracts (Plant / LLaVA / smoltalk / Negative / offline_qa). Unified record schema. v1 dummy-image trick vs v2 native `image=None` routing. |
| [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md) | `build_mix.sh` + `mix.py` orchestrator. Env-var-driven storage roots. Idempotence + determinism guarantees. |

### Production mix

| File | Covers |
|---|---|
| [`B-mix-50k-v2.md`](B-mix-50k-v2.md) | **Production canonical.** Current 50K mix (Plant 44 % / LLaVA 30 % / smoltalk 15 % / Negative 10 % + offline_qa persona). Per-bucket sizes, finetune-side wiring, reproduction command. |

## Reading order

1. [`01-data-prefix.md`](01-data-prefix.md) — the v4 camera-state gate. Concept + plumbing.
2. [`02-bucket-design.md`](02-bucket-design.md) — what each source bucket looks like.
3. [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md) — how the JSONL gets built.
4. [`B-mix-50k-v2.md`](B-mix-50k-v2.md) — current production config in detail.

## Test status

96 / 96 unit + integration tests green (`src/data_mix/tests/`). Tests
use mocked HF streams — no network required to run the suite.

## Related

| Location | Purpose |
|---|---|
| `src/data_mix/` | Code: builders, orchestrator, configs |
| `src/finetune/src/data.py` | Where the mix lands — `load_vision_dataset` + `build_vision_messages` + the prefix dispatch |
| `src/finetune/src/regularization.py` | KL + L2 implementation (finetune-side companion to the prefix gate) |
| [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md) | Full anti-forgetting design + the shipped recipe |
| [`../finetune/01-pipeline.md`](../finetune/01-pipeline.md) | Where the bf16 SFT'd merged model comes from |
