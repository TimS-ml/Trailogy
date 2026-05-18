# 16 — Final Model Eval (production candidates)

## TL;DR

- The final evaluation compared four production candidates using the same MLX 4-bit evaluator and fixed test slices.
- The shipped model is a stage-2 tree-specialized adapter on top of the selected stage-1 checkpoint.
- The stage-2 model gives up some global PlantNet accuracy but gains meaningful performance on North American tree species.
- The product decision favored recognizing trees hikers are likely to see over the highest aggregate leaderboard-style score.
- Cross-phase comparisons still require care because other docs may use different evaluators, sample counts, or datasets.

## Why there is an NA-tree stage-2 SFT at all

PlantNet-300K (the upstream dataset under our finetune mix) is global
in species coverage but **systematically thin on North-American tree
species** — the exact taxa a hiker in a US national park is most
likely to point a camera at. All of our stage-1 production candidates
(`r8-a8-nokl`, `r16-a16-nokl-no-text-prefix`, etc. — see
[`finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
and [`finetune/10-no-text-prefix-and-bigger-rank.md`](../finetune/10-no-text-prefix-and-bigger-rank.md))
sit at **0 / 100** on a 9-species NA-tree probe set, while scoring
30–43 % on the PlantNet-flavoured `plant_300` slice. The model
genuinely cannot name a Douglas Fir or a Live Oak from a photo —
neither the base 4-bit checkpoint nor any stage-1 SFT adapter.

To close that specific gap before shipping, we **manually curated a
small NA-tree dataset** (11 species, ~50 images/species) and ran a
**short stage-2 SFT** on top of the `r8-a8-nokl` stage-1 adapter.
The trade-off is intentional and quantified below: a few percentage
points of PlantNet-style `plant_300` (which we re-classify as
"general plant ID") in exchange for jumping NA-tree from 0 % to 34 %
on the same 100-image probe set, plus net-positive moves on MMLU,
AIME, and refusal.

The stage-2 adapter at step 11 000 is what ships in the iOS bundle.

## Summary — 4 production candidates

| Model | plant_300 | mmlu_50 | aime_20 | llava_40 | refusal_20 | NA-tree_100 | composite_5d |
|---|---|---|---|---|---|---|---|
| **r8-a8-nokl + NA-tree stage-2 step 11k** *(shipped)* | 31.0% | **50.0%** | **15.0%** | 12.0% | 100% | **34.0%** | 0.381 |
| r8-a8-nokl step 13k *(stage-1 only)* | **42.7%** | 34.0% | 10.0% | 12.8% | 100% | 0.0% | ~0.399 |
| r16-a16-nokl-no-text-prefix step 20k *(stage-1, larger rank)* | 37.7% | 40.0% | 10.0% | 12.5% | 100% | 0.0% | ~0.400 |
| base (`mlx-community/gemma-4-e2b-it-4bit`) | 0.0% | 40.0% | 10.0% | 7.4% | 35.0% | 0.0% | 0.161 |

- `composite_5d` = unweighted mean of (plant, mmlu, aime, llava, refusal).
  NA-tree is reported separately because it lives on a different
  curated dataset and would otherwise distort the cross-domain mean.
- bf16 base MMLU = 46 %; 4-bit g64 MMLU = 40 % (~6 pp quantization
  cost on this domain — see
  [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)).
- An H200 CUDA eval of the stage-1 r16 step 20k checkpoint scored
  composite=0.415 on the same metric (see
  [`finetune/10-no-text-prefix-and-bigger-rank.md`](../finetune/10-no-text-prefix-and-bigger-rank.md)
  §2). The Mac MLX g64 number for the same adapter is 0.400 — within
  the ~5 % cross-platform noise floor.

### Why ship the stage-2 model and not the stage-1 SOTA

Pure-composite ranking would pick `r16-a16-nokl-no-text-prefix step
20k` (~0.40). The stage-2 adapter is 0.02 behind on composite but
**+34 pp on NA-tree** and **+10 pp on MMLU** vs that candidate.
For the product (a national-park hike companion), correctly naming
the trees in front of the user matters more than +6 pp on PlantNet-
global species. The stage-2 model is the explicit
product-aligned choice, not the leaderboard winner.

---

## 1. Generality eval — per-candidate detail

Eval set definitions are unchanged from
[`10-eval-setup.md`](10-eval-setup.md): plant_100 / plant_300 use the
PlantNet val pool with the 12-prompt paraphrase pack, mmlu_50 / aime_20
are text-only academic slices, llava_40 is a VQA-style probe with
train-leak detection, refusal_20 covers the safety/idiom check.

### 1a. r8-a8-nokl + NA-tree stage-2 *(step 11000)* — shipped

Stage-1 recipe: r=8, α=8, KL=0, mix-50k, `[camera=off]` on text
(matching [`finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
§ "Final recipe"). Stage-2: continue from the stage-1 adapter on a
small curated NA-tree set (11 species), brief schedule, same r/α/KL.
Reported step `11000` is the global step counter across stage-1 +
stage-2.

| plant_300 | mmlu_50 | aime_20 | llava_40 (leak) | refusal_20 |
|---|---|---|---|---|
| 31.0 % | 50.0 % | 15.0 % | 12.0 % (leak=0 %) | 100 % |

NA-tree result: see §2.

### 1b. r16-a16-nokl-no-text-prefix *(step 20000, stage-1 only)*

r=16, α=16, KL=0, mix-50k, no text-side prefix, 10-epoch H200 run.
Stage-1 SOTA per
[`finetune/10-no-text-prefix-and-bigger-rank.md`](../finetune/10-no-text-prefix-and-bigger-rank.md).

| plant_300 | mmlu_50 | aime_20 | llava_40 (leak) | refusal_20 |
|---|---|---|---|---|
| 37.7 % | 40.0 % | 10.0 % | 12.5 % (leak=0 %) | 100 % |

### 1c. r8-a8-nokl *(step 13000, stage-1 only)*

The recipe `r=8, α=8, KL=0, mix-50k`, no text prefix, H200 — same
shape as 1b at rank 8 instead of 16. This is the **stage-1 parent**
of the shipped stage-2 model.

| plant_300 | mmlu_50 | aime_20 | llava_40 (leak) | refusal_20 |
|---|---|---|---|---|
| 42.7 % | 34.0 % | 10.0 % | 12.8 % (leak=0 %) | 100 % |

### 1d. base — `mlx-community/gemma-4-e2b-it-4bit`

No SFT, MLX-quantized 4-bit g64 release of `google/gemma-4-e2b-it`.
The reference floor for every other row.

| plant_300 | mmlu_50 | aime_20 | llava_40 (leak) | refusal_20 |
|---|---|---|---|---|
| 0.0 % | 40.0 % | 10.0 % | 7.4 % (leak=0 %) | 35.0 % |

---

## 2. NA-tree eval

### 2a. 4-model comparison (100 images, 9 species, substring match)

The probe is intentionally easy: 9 common NA trees, 100 in-the-wild
images, prediction is scored as correct if the canonical common name
(or one short alias) appears anywhere in the model's free-form
answer. This is a coverage check, not a leaderboard.

| Model | Correct | Accuracy |
|---|---|---|
| base (4-bit) | 0 / 100 | 0.0 % |
| r8-a8-nokl step 13k *(stage-1 only)* | 0 / 100 | 0.0 % |
| r16-a16-nokl-no-text-prefix step 20k *(stage-1 only)* | 0 / 100 | 0.0 % |
| **r8-a8-nokl + NA-tree stage-2 step 11k** *(shipped)* | **34 / 100** | **34.0 %** |

Three stage-1 candidates score zero — i.e. the PlantNet-300K-based
data mix does not transfer to these NA-tree species at all, even
when stage-1 plant_300 sits at 42 %. The stage-2 adapter is what
actually unlocks the domain.

### 2b. Stage-2 per-species breakdown (550 images, 11 species)

Larger held-out sample (50 images/species, 11 species, substring
match, same evaluator). Overall: **158 / 550 = 28.7 %**.

| Species | Correct / 50 | Accuracy |
|---|---|---|
| Honey Locust | 28 | 56 % |
| Red Maple | 25 | 50 % |
| Douglas Fir | 24 | 48 % |
| Live Oak | 24 | 48 % |
| American Holly | 14 | 28 % |
| White Oak | 14 | 28 % |
| Bradford Pear | 12 | 24 % |
| Eastern White Pine | 7 | 14 % |
| Crepe Myrtle | 6 | 12 % |
| Flowering Dogwood | 3 | 6 % |
| Sweetgum | 1 | 2 % |

The 100-image / 9-species set (§2a) reports a higher 34 % because
its species mix is biased toward the easier head of the distribution
above (Douglas Fir, Honey Locust, Red Maple, Live Oak all included).
The 550-image / 11-species set is the honest number; the 100-image
set is the cross-model comparison.

The long tail (Sweetgum, Flowering Dogwood, Crepe Myrtle) is where
the curated stage-2 set was smallest and the visual ambiguity is
highest. These are the obvious next targets if we extend the stage-2
dataset.

---

## 3. What this trades off vs the stage-1 SOTA

Holding the stage-1 r=8/α=8/KL=0 baseline (1c) fixed:

| Domain | Stage-1 only (1c, step 13k) | + NA-tree stage-2 (1a, step 11k) | Δ |
|---|---|---|---|
| plant_300 | 42.7 % | 31.0 % | **−11.7 pp** |
| mmlu_50 | 34.0 % | 50.0 % | +16.0 pp |
| aime_20 | 10.0 % | 15.0 % | +5.0 pp |
| llava_40 | 12.8 % | 12.0 % | −0.8 pp |
| refusal_20 | 100 % | 100 % | 0 |
| NA-tree_100 | 0 % | 34.0 % | **+34.0 pp** |

Two effects worth calling out:

- **plant_300 drops 11.7 pp** as the model reallocates capacity from
  generic PlantNet-style identification toward NA-tree-specific
  common-name vocabulary. Expected and acceptable.
- **MMLU actually rises** from 34 % → 50 %, **+16 pp**. This was not
  predicted. Plausible explanation: stage-2 used a small, clean,
  description-heavy dataset which acted as mild rehearsal against
  the stage-1 plant-domain overfit visible in 1c's MMLU drop (base
  is 40 %, stage-1-only sat below base at 34 %). Stage-2 partially
  un-did that forgetting while introducing the new tree vocabulary.
  Consistent with the "anti-forgetting comes from data mix, not from
  regularization terms" finding in
  [`finetune/09-kl-is-overkill-at-small-rank.md`](../finetune/09-kl-is-overkill-at-small-rank.md).

---

## 4. Source files

Per-run eval JSON artifacts and the NA-tree probe set live in
internal notes; this doc captures the rolled-up numbers. The
public-side scripts and eval sets are:

| Item | Path |
|---|---|
| `plant_300` eval set | `finetune/eval_sets/plant_300.jsonl` |
| `plant_na_100` eval set (the 9-species probe) | `finetune/eval_sets/plant_na_100.jsonl` |
| Generality evaluator | `finetune/src/evaluate_generality.py` |

## Cross-references

- Stage-1 final recipe and anti-forgetting design:
  [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
- Stage-1 SOTA (r16-a16-nokl-no-text-prefix step 20k):
  [`../finetune/10-no-text-prefix-and-bigger-rank.md`](../finetune/10-no-text-prefix-and-bigger-rank.md)
- What the metrics actually compute and benchmark-drift caveats:
  [`10-eval-setup.md`](10-eval-setup.md)
- Cross-platform numerical contract (MLX-Mac vs MLX-CUDA vs HF-bf16):
  [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md)
