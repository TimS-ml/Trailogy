# 09 — KL is overkill on small-rank LoRA; it actively blocks plant learning

## TL;DR

- This doc tests whether KL regularization helps a small-rank LoRA model learn plant species while preserving general ability.
- The sweep found that KL blocked plant learning at small rank: with KL 0.05, ranks 4 through 32 stayed at 0.000 plant match.
- The best recorded small-rank recipe in this doc used rank 8, alpha/r 1.0, and no KL, reaching 0.23 plant match while staying above the base MMLU score.
- The conclusion is that data mix and conservative adapter settings did more for anti-forgetting than KL on this task.

## 1. KL Strength vs the (plant, mmlu) Dual Indicator

`PLANT_IMAGE_ROOT=data/english-desc-v2/images_resized/test`,
`base_model=unsloth/gemma-4-E2B-it` (base mmlu=0.460, aime=0.100,
plant=0.000).

### Rank fixed at r=8 (the SOTA window), KL sweep:

| run | α/r | KL weight | step | plant | mmlu | aime | composite |
|---|---|---|---|---|---|---|---|
| r8-a16-kl0001 (H200) | 2.0 | 0.0001 | 1000 | 0.020 | **0.460** | 0.100 | 0.208 |
| r8-a16-nokl (H200)   | 2.0 | 0      | 1000 | 0.040 | 0.280 | 0.200 | 0.169 |
| r8-a16-nokl (H200)   | 2.0 | 0      | 2000 | 0.020 | 0.260 | 0.100 | 0.131 |
| **r8-a8-nokl (SOTA, H200)** | **1.0** | **0** | **4647** | **0.230** | **0.480** | **0.200** | **0.319** |
| r8-a16-mix50k (4090) | 2.0 | 0      | 4000 | 0.140 | 0.560 | 0.150 | 0.243 |
| r8-a16-mix50k (4090) | 2.0 | 0      | 5000 | 0.150 | **0.620** | 0.200 | **0.380** |
| r8-a16-mix50k (4090) | 2.0 | 0      | 6000 | 0.150 | 0.580 | 0.200 | 0.372 |

Observation 1: **KL=0.0001 = "nothing was learned"**. mmlu identical
to base (0.460), aime identical to base (0.100), plant barely moves
(0.02 vs base=0.00 is ROUGE noise). 1000 steps of training = no
training.

Observation 2: **KL=0 but α/r=2.0 = "collapse"**. From step 1000 to
2000 every metric drops (plant 0.04 → 0.02, mmlu 0.28 → 0.26, aime
0.20 → 0.10). This tracks the docs/08 finding that α/r=2.0 forward
gain is too aggressive, and is unrelated to KL.

Observation 3: **KL=0 and α/r=1.0 = SOTA**. After 4647 steps,
plant=0.23, mmlu=0.48 (above base!), aime=0.20 (2× base).

Observation 3b: **KL=0 and α/r=2.0 + mix-50k (4090 long run) also
holds**. The 4090 `r8-a16-drop005-mix50k` runs to step 6000
(epoch 1.94), with mmlu rising from 0.56 at step 4k to a peak of
0.62 at step 5k, settling back to 0.58 at step 6k — **all above
the base 0.46 the whole way**. This run has the same α/r=2.0 and
KL=0 as the H200 `r8-a16-nokl`, but because S_step=0.0064
(lr=2e-4, eff_bs=16) and mix-50k carries a 30 % non-plant bucket
that supplies dense anti-forgetting signal, mmlu didn't collapse —
it improved. This is the strongest longitudinal validation of
§3(c) "real anti-forgetting comes from the data mix": stable
across 3 checkpoints (4k/5k/6k). Plant plateaus at 0.14–0.15;
pushing it further needs more epochs or vision-tower unfreezing.

### Rank sweep, KL fixed at standard strength (0.05):

| run | rank | α | α/r | step | plant | mmlu | aime |
|---|---|---|---|---|---|---|---|
| r4-a8-kl005     | 4   | 8   | 2.0 | 1000 | 0.000 | 0.540 | 0.150 |
| r4-a8-kl003     | 4   | 8   | 2.0 | 4000 | 0.000 | 0.520 | 0.050 |
| r16-a16-kl005   | 16  | 16  | 1.0 | 1000 | 0.000 | 0.480 | 0.150 |
| r32-a32-kl005   | 32  | 32  | 1.0 | 1000 | 0.000 | 0.520 | 0.150 |
| r32-a32-kl005   | 32  | 32  | 1.0 | 3000 | 0.000 | 0.520 | 0.050 |
| **r256-a256-kl005** | **256** | 256 | 1.0 | 1000 | 0.000 | **0.100** | 0.050 |

Observation 4: **With KL on, plant=0 for every rank from r=4 to r=32**.
This is not a capacity problem — r=32 LoRA already gives ~3.5 M
trainable params, far more than the hidden representation needed for
1000-class PlantNet. **KL pulls the language-side logit distribution
back toward the base, and the base never says a specific species name
for a plant image**.

Observation 5: **KL=0.05 at r=256 fails outright, mmlu collapses to
0.10**. The intuition "KL ⇒ mmlu is preserved" only holds at small
rank. Once the LoRA subspace dimensionality grows, token-level KL
divergence is insufficient to pull a high-dimensional weight shift
back into place.

## 2. Why KL kills plant learning

The PlantNet ground-truth completion looks like:

```
[camera=on] What is this plant?
→ This is Quercus robur (English oak). It's a deciduous tree ...
```

The base model on the same image produces something like:

```
→ I can see a tree with green leaves. The leaves appear to be ...
       I cannot identify the specific species without more context.
```

The base will never output "This is Quercus robur". **This is the
core problem with the KL objective**:

`KL(p_sft || p_base)` is computed per-token in logit space. For the
SFT model to say "Quercus" after "This is ", it must place
probability on a token that has near-zero probability under the base
(the base almost always says "a tree" or "a plant" rather than a
specific species token). Every learning step has to fight the KL
gradient, which pushes directly toward "say something generic".

Empirically the two-sided gradient magnitudes are uneven:

- **CE loss (plant)**: ~0.5–1.0 nat / token per batch on average,
  pointing toward the species-name token.
- **KL loss (kl_weight=0.05)**: ~0.8–2.5 per batch on average
  (from `reg_kl` in metrics.jsonl), pointing toward the base
  model's generic-language distribution.

`reg_kl` is the same order of magnitude as plant CE, or even larger.
0.05 × reg_kl ≈ 0.04–0.13 nat / token — comparable to plant CE. So
**the plant-learning gradient signal is cancelled out by an equal-
magnitude opposite force**.

Dropping KL to 0.0001 (r8-a16-kl0001): 0.0001 × reg_kl ≈
0.0001–0.0003 nat / token, far below plant CE → should be
learnable. But empirically mmlu didn't move, aime didn't move, and
plant barely moved — meaning 1000 steps at this lr (2e-4) + batch
(16) setting was not enough total update magnitude for anything to
emerge, independent of KL strength. This run is "training had not
yet kicked in" rather than evidence that KL=0.0001 is a SOTA
candidate; a longer run is needed before drawing a conclusion.

## 3. Counter-examples to "KL = catastrophic-forgetting defense"

The original argument in docs/07 §1:

> "KL penalty against the base model's outputs (RLHF-style
> anti-drift applied at SFT time)."

assumes "drift away from base = forgetting". In practice:

**(a) drift is not the same as forgetting**

r8-a8-nokl trains to mmlu=0.48, +2 % above the base 0.46. Even when
LoRA updates are completely unconstrained, as long as the rank stays
small, **the learned representation can actually compensate for the
base's weak spots on shared tasks**. Much of what gets called "drift
away from base" is in fact "drift toward better behaviour on shared
tasks".

**(b) anchoring to base does not preserve generality**

r256-a256-kl005 has KL=0.05 turned on and still drops mmlu to 0.10.
KL anchoring fails at high rank — not because KL was set too low
(0.05 is standard RLHF strength), but because KL only anchors the
token distribution, not the displacement direction of a high-
dimensional LoRA subspace. At r=256 there are ~58 M trainable
parameters; the KL counter-gradient is diluted across them.

**(c) the real anti-forgetting force comes from the data mix**

mix-50k contains 30 % image_other, 15 % text (including smoltalk QA),
and 10 % negative. Each batch sees non-plant samples that contribute
a "preserve the original capability" gradient. **This is a dense
anti-forgetting signal; KL is sparse (one logit-divergence term per
batch)**.

Empirically: r8-a8-nokl runs with KL fully off, and because of the
mix buckets, mmlu doesn't drop — it rises. The 4090
`r8-a16-drop005-mix50k` (also KL=0) at checkpoints 4k/5k/6k shows
mmlu = 0.56/0.62/0.58, **all above the base 0.46**, with
`eval_nonplant_loss` falling monotonically 1.483 → 1.476 → 1.471
(see docs/08 §A2.2). That is longitudinal validation across 3
checkpoints: the mix-bucket lever alone is sufficient. **KL becomes
a redundant second lever, and its cost is killing plant learning**.

## 4. Where can KL still help?

KL is not categorically wrong; the scope just needs to shrink:

- **High-rank LoRA + in-distribution finetuning** (e.g. instruction
  tuning that doesn't introduce a new vocabulary): KL can still
  contain logit drift. But when small rank + out-of-distribution
  (plant species names are OOD tokens) are both present, KL is too
  much.
- **Data without an anti-forgetting bucket**: if the train set is
  100 % PlantNet, KL is necessary — there's no mmlu / negative / qa
  bucket providing dense gradient per batch, so KL is the last line
  of defence. Our mix-50k already builds the dense line; KL becomes
  over-defence.
- **The RLHF stage**: KL is a well-studied tool for preserving
  fluency under reward-model feedback. That's a different setup
  from SFT fitting plant-labelled data directly.

## 5. KL-rank matrix to test next

Current grid (✓ = measured, ? = not measured):

| KL ↓ \\ rank → | r=4 | r=8  | r=16 | r=32 | r=256 |
|---|---|---|---|---|---|
| 0       | ?   | **SOTA (a8)** ✓ | ? | ? | ?     |
| 0.0001  | ?   | ✓ (kl0001) under-trained | ? | ? | ?     |
| 0.005   | ?   | ?    | ?    | ?    | ?     |
| 0.03    | ✓ (a8) | ? | ?    | ?    | ?     |
| 0.05    | ✓ (a8) | ?   | ✓ (a16) | ✓ (a32) | ✓ — mmlu collapsed |

Key follow-ups:

1. **Can a smaller rank than r8-a8-nokl still learn plant?** Run
   r4-a4-nokl and check whether plant collapses to 0. If it does,
   **rank is the lower bound for plant learning, decoupled from KL**.
2. **Does r8-a8 with KL=0.005 (one notch down) preserve mmlu without
   choking plant?** This is the search for "KL=0 SOTA + a touch of
   KL = a bit more mmlu margin".
3. **r8-a8-nokl + vision-tower last-2**: can plant go from 0.23 to
   0.40+? Decoupled from KL; this is the docs/08 §7 next-step item.

## 6. Relationship to docs/07 (anti-forgetting regularization)

docs/07 frames KL + L2 + camera prefix as three complementary levers.
**That framing is still correct**, but:

- The marginal contribution of the three levers is **unequal**.
- In the current setup (mix-50k + camera-prefix-v4 + small-rank
  LoRA), **KL has the smallest marginal benefit** and a non-trivial
  side effect.
- **L2 (weight anchor toward init) was never properly tested** —
  `l2_enabled` is false in every cloud_sweep config. The docs/07
  §2 lever has not actually been exercised. If future SFT introduces
  a more distribution-distant task, L2 may become a cheaper anchor
  than KL — it anchors in parameter space, not logit space, and is
  agnostic to whether the new tokens (plant species names) are
  OOD.
- The camera prefix (v4) is a dense signal that surfaces modality
  information at the logit level on every batch and effectively
  replaces part of KL's job. When `[camera=on]` is present, the
  model can "switch speaking style" — **that itself is
  anti-forgetting**.

So docs/07's overall design (defence in depth) holds; it's just that
the KL layer becomes redundant under small-rank + camera-prefix +
mix-50k.

## 7. Action items

1. **Default KL=0 in the next sweep**. Convert every still-queued
   `*-kl005` / `*-kl003` config in cloud_sweep into a `*-nokl`
   variant (and sweep α/r=1.0 across the board). The existing data
   already supports this conclusion.
2. **Commit the r8-a8-nokl yaml to the repo**. Currently it has to
   be reverse-engineered from train.log; the recipe is not in git,
   which means the production recipe cannot be reproduced from the
   repo alone (see docs/08 §6 for the full yaml).
3. **Keep L2 anchor as a fallback**: if a future task introduces
   mmlu drift, L2 (weight-space anchor) is more likely than KL
   (logit-space anchor) to work at small rank. The implementation
   already exists in `src/finetune/src/regularization.py`; just
   set `l2_enabled: true`, `l2_weight: 1e-4` in the yaml to try.
4. **Run r4-a4-nokl to find the rank lower bound**: confirm whether
   plant learning starts at r=8 only. If r=4 also learns plant, the
   nokl recipe can be made even lighter; if r=4 doesn't learn
   plant, r=8 is confirmed as the SOTA sweet spot.

## 8. One-sentence summary

In the current SFT setup (mix-50k + camera-prefix-v4 + small-rank
LoRA), **the KL anchor pins the model on the base behaviour of "do
not name the species"**, which is the single largest obstacle to
plant learning. The SOTA recipe is r=8, α=8, **KL=0**, run to a full
3 epochs (~4600 steps). Anti-forgetting is delivered by the data
mix; KL is not required.
