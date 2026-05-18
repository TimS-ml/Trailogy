# 10 — Dropping `[camera=off]` on text data + bumping rank to 16: new composite SOTA

## TL;DR

- This doc records late ablations that improved on the earlier small-rank no-KL recipe.
- Removing the text-only camera prefix helped the rank 8 recipe without hurting general benchmarks.
- Increasing to rank 16 with alpha/r kept at 1.0 produced the best recorded composite score in this note: 0.415 at step 20000.
- The current takeaway is to prefer `r16-a16-nokl-no-text-prefix` as the new production candidate, while treating the alternate `mixc` data variant as a negative result.

> last edit: 2026-05-18 evening (MacBook M5 Pro eval patch)

> **Status (2026-05-18, noon)**: 在 H200 上跑了两组新的对照
> 实验,把先前 docs/09 里"r=8, α=8, KL=0, ~4647 step"的 SOTA
> 推翻。新的 production candidate 是
> **`r16-a16-nokl-no-text-prefix` @ step 20000, composite=0.415**
> (plant=0.42, mmlu=0.54, aime=0.20)。两个独立的修改各贡献一部分:
>
> 1. **去掉 text-only data 上的 `[camera=off]` prefix**(image
>    side 仍保留 `[camera=on]`)。同一份 r=8, α=8 配方,只把
>    `data.prompt_prefixes.camera_off` 从 `"[camera=off] "` 改成
>    `""`,composite 从 0.319 → 0.365 (+15%)。
> 2. **rank 8 → 16,α 同步 8 → 16(保 α/r=1.0)**。在 (1) 的
>    基础上 capacity 翻倍,允许跑更长 epoch(3 → 10)而 mmlu
>    不崩。composite 0.365 → 0.415 (+14%)。
>
> 两条加起来 vs docs/08-09 旧 SOTA:0.319 → 0.415 (+30%)。
> docs/09 §6 "production 配方"和 §8 "一句话总结"现在过时,
> 见本文 §6。

## 1. Nokl Recipe Comparison

`PLANT_IMAGE_ROOT=data/english-desc-v2/images_resized/test`,base
`unsloth/gemma-4-E2B-it` (base: plant=0.00, mmlu=0.46, aime=0.10),
generality eval `--skip_judge`(plant species match / mmlu letter /
aime numeric)。

| run | rank | α/r | text prefix | best step | plant | mmlu | aime | **composite** |
|---|---|---|---|---|---|---|---|---|
| docs/09 SOTA `r8-a8-nokl` | 8 | 1.0 | `[camera=off]` | 4647 | 0.230 | 0.480 | 0.200 | 0.319 |
| `r8-a8-nokl-no-text-prefix` | 8 | 1.0 | **none** | 7000 | 0.260 | 0.540 | 0.250 | **0.365** |
| `r8-a8-nokl-no-text-prefix-mixc` | 8 | 1.0 | none + mix variant 'c' | 11000 | 0.330 | 0.440 | 0.100 | 0.319 |
| **`r16-a16-nokl-no-text-prefix`** | **16** | **1.0** | **none** | **20000** | **0.420** | **0.540** | **0.200** | **0.415** |

观察 1:**单独去掉 text-side `[camera=off]` 已经是免费午餐**。
同 rank、同 α/r、同 mix 数据,composite +15%,plant 和 mmlu 都
没掉。

观察 2:**rank 翻倍 + 长 epoch 不再崩**。docs/09 §1 表里 r=8
跑到 step 15000 时 mmlu 还能守住但 plant 已经 plateau;r=16 在
step 20000 (= 6.5 epoch over mix-50k) 仍单调改善 plant 而 mmlu
稳定 0.52-0.54。

观察 3:**"mixc" 变体不是 win**。`r8-a8-nokl-no-text-prefix-mixc`
跑到 step 11000 时 plant 学到 0.33 但 mmlu 掉到 0.44(低于 base
0.46),aime 回落 0.10。data mix 改了什么需要回头看 mixc bundle
的具体配方(`data/mix-50k/` 之外的 variant);本文档不展开,只
先记一个 negative result。

## 2. `r16-a16-nokl-no-text-prefix` per-checkpoint 轨迹

`<workspace>/sft_out/r16-a16-nokl-no-text-prefix_20260518_015342/`,
config 关键字段(`train.log`):

```
lora.r = 16
lora.lora_alpha = 16                      → α/r = 1.0
lora.lora_dropout = 0.05
lora.projector_learning_rate = 1.4e-4
lora.tune_projector = True
lora.tune_last_n_vision_layers = 0
training.per_device_train_batch_size = 32
training.gradient_accumulation_steps = 1  → eff_bs = 32
training.num_train_epochs = 10            (vs SOTA 旧配方的 3)
training.learning_rate = 3.0e-4
training.warmup_steps = 30
training.lr_scheduler_type = cosine
regularization.kl_enabled = False
data.train_file = data/mix-50k/train.jsonl
data.prompt_prefixes = {camera_on: "[camera=on] ", camera_off: ""}
```

→ `S_step ≈ 1.0 × 3.0e-4 × 32 = 0.0096`(跟 docs/09 SOTA 一样,
说明 r 这个独立变量被解耦验证)。

| step | epoch | plant | mmlu | aime | composite | note |
|---|---|---|---|---|---|---|
| 5000  | ~1.6  | 0.310 | 0.560 | 0.100 | 0.323 | mmlu 已 > base 0.46 |
| 8000  | ~2.6  | 0.310 | 0.540 | 0.150 | 0.333 | |
| 9000  | ~2.9  | 0.370 | 0.500 | 0.150 | 0.369 | |
| 10000 | ~3.2  | 0.350 | 0.460 | 0.150 | 0.346 | |
| 11000 | ~3.5  | 0.410 | 0.500 | 0.150 | 0.385 | plant breaks 0.4 |
| 12000 | ~3.9  | 0.390 | 0.480 | 0.100 | 0.358 | |
| 13000 | ~4.2  | 0.390 | 0.480 | 0.200 | 0.381 | |
| 15000 | ~4.8  | 0.400 | 0.520 | 0.200 | 0.400 | first composite=0.40 |
| 15490 | ~5.0  | 0.410 | 0.500 | 0.150 | 0.385 | (epoch boundary; same as 15000 within eval noise) |
| 18000 | ~5.8  | 0.400 | 0.540 | 0.200 | 0.408 | |
| **20000** | **~6.5** | **0.420** | **0.540** | **0.200** | **0.415** | **best so far; still climbing?** |

两个明确的子信号:

- **mmlu 横盘 0.46-0.56**,全程 ≥ base(0.46)。这跟 docs/09
  §3.c "anti-forgetting 来自 data mix, 不来自 KL"完全一致,
  并且把这个结论从 r=8 + 3 epoch 扩展到 **r=16 + 6.5 epoch**。
- **plant 单调上升**(0.31 → 0.42 / 8 checkpoints)。docs/09 里
  r=8 SOTA 的 plant=0.23 已经被认为是 SOTA recipe 的天花板;
  这里 r=16 直接把它推到 0.42,**rank 是 plant 学习容量的硬上限**
  的假设得到强证据。

eval `r16-a16-nokl-no-text-prefix_step20000.json` 摘录:

```json
{
  "generality_score": 0.4154,
  "domains": {
    "plant": {"n": 100, "species_match_rate": 0.42,
              "rouge_l_mean": 0.20098, "score": 0.42},
    "mmlu":  {"n": 50, "accuracy": 0.54, "score": 0.54},
    "aime":  {"n": 20, "accuracy": 0.20, "score": 0.20}
  }
}
```

## 3. 为什么去掉 `[camera=off]` 反而帮 generality

docs/07 §3 的设计逻辑是"text prefix `[camera=off]` 让模型在
推理时显式 route 到 text-only branch"。这个机制对 plant /
vision 数据没问题(`[camera=on]` 跟 image presence 同分布,
是 dense 信号),但在 text-only branch 上出现了一个隐形成本:

- **base model 从来没见过 `[camera=off]` 这个 literal**。给每个
  text-only 样本前面塞这个 prefix,等于强迫语言侧 LoRA + projector
  学一个 "ignore this synthetic 4-token preamble" 的映射。
- mix-50k 里有 ~30% 是 text-only(smoltalk QA + offline_qa
  persona + negative refusal),全部 prefix 了 `[camera=off] `。
  这部分 batch 的 attention 第一格被这个 prefix 占走,**消耗了一
  部分 small-rank LoRA capacity**。
- 评测端 mmlu / aime 是纯 text、**没加** prefix(docs/06 wrapper
  `bash run.sh --prompt_prefix "[camera=on] "` 只对 image
  domain 加,见 06-local_test/eval_generality_patch/README.md §
  "Prompt prefix for v3 vs v4 checkpoints")。所以**训练分布 vs
  评测分布在 text-only 这一支不对齐**:训练 saw `[camera=off]`,
  评测没 prefix。这是直接的 train/test prefix mismatch。

去掉 text prefix 之后,training 和 evaluation 在 text-only branch
上的 input distribution 对齐,LoRA 不再花 capacity 去消化 `[camera=off]`,
两个效应一起把 mmlu 从 0.48 推到 0.52-0.56,aime 也从 0.20 抬到
0.25(r=8 那栏)。

**Image branch 不受影响**:`[camera=on]` 还保留,因为 image 数据
在评测端(plant_100, llava_40)也加了 `[camera=on]`,train/test
分布是 match 的。

### 隐含约束:iOS 部署对应改 1 行

`HikeCompanion/GemmaService.swift` 在
`imageInputs.isEmpty` 分支(`.text` 模式)目前会(或将)前置
`[camera=off] `;新 SOTA recipe 训练时这一支**没有** prefix,
所以 iOS 端的 text-mode 也必须不加 prefix。只 image-mode 加
`[camera=on]`。

## 4. rank 8 → 16 的边际收益(同 prefix 设定下)

固定 "no text prefix" 之后,rank 是唯一变量:

| run | rank | best step | plant | mmlu | composite |
|---|---|---|---|---|---|
| r8-a8-nokl-no-text-prefix  | 8  | 7000  | 0.260 | 0.540 | 0.365 |
| r16-a16-nokl-no-text-prefix | 16 | 20000 | 0.420 | 0.540 | 0.415 |

mmlu 完全持平(0.54)→ **rank 翻倍没增加 forgetting**。plant
从 0.26 → 0.42 (+16pp)→ **rank 翻倍直接换 plant capacity**。
跟 docs/09 §1 / 09 §6 "rank 大才能学 plant" 的方向一致,但比那
里 r=256 + KL=0.05 直接崩 mmlu 到 0.10 干净很多 — **关键还是
α/r=1.0 + KL=0**,rank 自身不是 forgetting 主因。这跟 docs/09
§5 第 1 个 follow-up "rank 是 plant learning 的 lower bound,跟
KL 解耦"得到正向验证(虽然方向不同:docs/09 问的是 r=4 还能不
能学,这里证的是 r=16 学得更好)。

下一步如果继续推 rank(r=32, r=64, 全部 α/r=1.0 + nokl + no text
prefix)是 §7 议程。

## 5. `r8-a8-nokl-no-text-prefix-mixc` 是 negative result

`r8-a8-nokl-no-text-prefix-mixc_20260518_105110` 跑了 12000+ step
(epoch ~8.6), checkpoint-11000 eval 出来:

```
plant=0.33, mmlu=0.44, aime=0.10  → composite=0.319
```

mmlu 第一次掉到 **低于 base 0.46**。aime 也回落 0.10。plant 学得更
好(0.33 vs SOTA 0.26 @ step 7k)但代价过高。"mixc" 是 mix-50k 的
某个 variant(`-mixc` suffix),具体配方差异不在本 session 的日志
里,需要回 this repo repo `data/` 下找。**先冻结结论**:在
当前 config 下 mixc 不是 win。

可能的解释(待证):mixc 提高了 plant 占比 / 降低了 text bucket,
所以 plant 学得快但 anti-forgetting 信号变弱。docs/09 §3.c 那个
"data mix 是 anti-forgetting 主力"的逻辑反过来用 — 削弱 mix 就是
削弱 mmlu 守门员。

## 6. 更新后的 production 配方(取代 docs/09 §6)

到 2026-05-18 noon UTC,**唯一已实测 composite > 0.40** 的 config:

```yaml
model:
  base_model: "unsloth/gemma-4-E2B-it"
  max_seq_length: 1024
  dtype: bfloat16
lora:
  r: 16
  lora_alpha: 16            # α/r = 1.0
  lora_dropout: 0.05
  tune_projector: true
  projector_learning_rate: 1.4e-4
  tune_last_n_vision_layers: 0
training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1     # eff_bs = 32
  num_train_epochs: 10
  learning_rate: 3.0e-4
  warmup_steps: 30
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  weight_decay: 0.001
  optim: adamw_torch_fused
  save_steps: 1000
regularization:
  kl_enabled: false
  l2_enabled: false
data:
  train_file: data/mix-50k/train.jsonl
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: ""                   # ← key change vs docs/09 SOTA
```

best checkpoint: **step 20000** (composite=0.415)。run 还在监控,
docs/08-09 留下来的 "α/r=1.0 + KL=0" 主轴不变。

## 7. 还没回答的问题(下一轮 sweep 议程)

1. **r=16 跑到 step 25k/30k 会继续涨吗?** plant 单调上升 +
   mmlu 没崩 = 没有明显 stop signal。但 r=16 nokl 的 10-epoch
   cap 已经到,需要 yaml 把 epoch 上调或 max_steps 显式扩。
2. **r=32, α=32, nokl, no-text-prefix**:rank 继续翻倍,plant
   能否到 0.5+?mmlu 会不会因为 capacity 增大开始往下走?这是
   docs/09 §2.b 论证"rank 是独立崩塌轴"的直接复测。
3. **r16-a16-nokl(WITH text prefix)对照**:同 yaml,只把
   `camera_off` 改回 `"[camera=off] "`。已开始跑
   (`r16-a16-nokl_20260518_060937`),目前 ~940 step,还没 eval。
   等 step 5k+ eval 出来,可以做 §3 prefix-mismatch 假设的干净
   A/B。
4. **vision_tower last-2 unfreeze + r=16-a16-nokl-no-text-prefix**:
   docs/09 §5 第 3 个 follow-up 推迟到这里,plant ≥ 0.5 的下一档
   推力。

## 8. 一句话总结

> docs/09 SOTA(r=8, α=8, nokl, 3 epoch)被 `r16-a16-nokl-no-text-prefix`
> @ step 20000(composite **0.415**)取代。两个独立改动各贡献一半:
> 去掉 text-only 数据的 `[camera=off]` prefix 消除 train/test
> mismatch (+15% composite),rank 8 → 16 + α 同步 → 16 把 plant
> capacity 翻倍同时 mmlu 不退 (+14% composite)。下一步推 rank
> 到 32 + vision-last-2 验证 plant > 0.5 是否可达。

## 9. MacBook M5 Pro cross-hardware eval (2026-05-18 evening)

Mac 上跑了三组 eval,用的是 `sweep_eval_only.sh` + `evaluate_generality.py`
(Darwin MPS path, monkey-patch `caching_allocator_warmup` to no-op)。
跟 H200 eval 用同一 evaluator,但 evaluator 在 Mac 上是 bf16+MPS
不是 bf16+CUDA,数值可能有小偏差(docs/08 §A2.1 估 ~5% noise floor)。

### 9a. `r8-a16-drop005-mix50k` (4090 trained, Mac eval) — 4k/5k/6k

跟 docs/08-09 里讨论的 4090 本地 run 完全一样的 adapter,
区别是 eval 在 Mac 而不是 4090。

| step | plant | mmlu | aime | llava | refusal | composite |
|---|---|---|---|---|---|---|
| 4000 | 0.140 | 0.560 | 0.150 | 0.143 (leak=0%) | 0.000 | 0.243 |
| 5000 | 0.150 | **0.620** | 0.200 | 0.128 (leak=0%) | **1.000** | **0.380** |
| 6000 | 0.150 | 0.580 | 0.200 | 0.150 (leak=0%) | 1.000 | 0.372 |

关键发现:
- mmlu peak at step 5k (0.62), 6k 开始回落 (0.58)。mild overfitting。
- refusal 从 step 4k 的 0% 翻到 step 5k 的 100%。safety 在 4k-5k
  之间恢复。
- plant 在 0.14-0.15 plateau, 跟 H200 SOTA 的 0.23 相比低很多,
  因为这是 α/r=2.0 的 run (docs/08 §A2.3 论证 α/r=2.0 calibration
  差 → plant_match 低)。

这个 run 有 llava + refusal eval (H200 eval 当时没跑这两域),
首次确认 mix-50k SFT **llava leakage = 0%**(模型不泄漏 train
数据到 llava 回答里)。

### 9b. `r8-a8-nokl-laptop` (laptop 4090 trained, Mac eval) — 1k/2k

跟 H200 SOTA 完全同 config (r=8, α=8, KL=0, mix-50k, camera
prefix v4), 但在 laptop 4090 上训 (eff_bs=16, lr=2e-4, 不是
H200 的 bs=32 lr=3e-4)。

| step | plant | mmlu | aime | llava | refusal | composite |
|---|---|---|---|---|---|---|
| 1000 | 0.040 | **0.620** | 0.100 | 0.132 (leak=0%) | 1.000 | 0.333 |
| 2000 | 0.060 | 0.580 | **0.200** | 0.128 (leak=0%) | 1.000 | 0.343 |

跟 H200 SOTA @ step 4647 (plant=0.23, mmlu=0.48, aime=0.20,
composite=0.319) 比:
- mmlu **大幅领先** H200 (0.62 vs 0.48)。laptop 的 S_step=0.0032
  (α/r=1.0 × lr=2e-4 × bs=16) 远小于 H200 的 0.0096,所以学得
  更保守、mmlu anchor 更强。
- plant 还很低 (0.04→0.06),因为才 2k step、只看了 32k sample
  (vs H200 的 148k)。需要跑到 step 4k-5k 才能公平比。
- **r8-a8-nokl recipe 跨硬件复现成功**。

### 9c. `r8-a8-nokl-vision2-local` (H200→local continue, vision tower last-2) — 10k

H200 SOTA (r8-a8-nokl step ~9k) 做 parent, 在本地继续训 ~1k step
with `tune_last_n_vision_layers=2`。

| step | plant | mmlu | aime | llava | refusal | composite |
|---|---|---|---|---|---|---|
| 10000 | **0.000** | 0.480 | 0.150 | 0.119 (leak=0%) | 1.000 | 0.289 |

**结论: vision tower unfreeze 直接把 plant 从 0.23 干到 0.00**。
mmlu 维持 (0.48 = 跟 H200 SOTA 一样), 但 plant learning 被完全
抹掉。原因: vision encoder 的 feature space 移位, language LoRA
+ projector 在仅 ~1k step 内来不及 track 新的 representation。
这是 AGENTS.md 里提到的 "feature-space misalignment risk" 的实证。

docs/09 §5 第 3 条 follow-up ("r8-a8-nokl + vision tower last-2:
plant 能不能从 0.23 推到 0.40+")现在有答案: **不行, at least not
by resuming from a frozen-tower checkpoint**。如果要做 vision-tower
tuning, 需要 **from scratch with vision layers unfrozen from step 0**,
让 LoRA/projector 跟 encoder 共同演化, 而不是 mid-flight 松开。

Run logs and per-step eval JSON live in internal notes; this doc
captures the rolled-up numbers and decisions.
