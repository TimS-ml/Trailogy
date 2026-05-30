# ViT baseline + new VLM backbones

Two related experiments on `feature/vit-baseline`, branched off
`feature/sft-quant-datamix`:

1. A **pure-vision ViT baseline** — replace the whole VLM with a ViT encoder
   plus a linear classification head, to measure how much plant-ID signal the
   vision encoder carries before any language decoding.
2. Adding **Qwen3.5-4B** and **InternVL3.5-4B** as alternative SFT backbones,
   both of which carry a full, trainable vision tower (unlike the shipped
   Gemma 4 E2B recipe, where the vision tower is frozen by default).

The ViT baseline is the first thing we train; the new-backbone work is staged
behind it.

## 1. ViT baseline (implemented)

Code lives in `src/finetune/vit_baseline/` (see its `README.md`). Pure-vision
classifier on the iNaturalist NA-Plantae ImageFolder (799 train classes,
~52 k train / ~6.6 k val images). Default backbone is `vit_base_patch16_siglip_224`
(same SigLIP family as the Gemma 4 vision tower), full bf16 SFT of the encoder
with a linear head; a frozen-backbone linear-probe config is also provided.

The val top-1 / top-5 is read alongside the VLM plant score from
`eval/evaluate_generality.py` — the gap is the language model's contribution on
top of the encoder.

## 2. New VLM backbones (staged)

Both target models were confirmed multimodal:

| Model | HF repo | arch / `model_type` | Notes |
|---|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | native `vision_config` + `text_config`, `image_token_id`; loads via `AutoModelForImageTextToText` |
| InternVL3.5-4B | `OpenGVLab/InternVL3_5-4B` | `InternVLChatModel` / `internvl_chat` | custom code (`auto_map`, needs `trust_remote_code`), dynamic image tiling, built-in `use_backbone_lora` / `use_llm_lora` flags |

### Open design points (resolve before implementing)

- **Trainer**: the current SFT loop is unsloth-specific and Gemma-shaped.
  Neither target is an unsloth-supported family, so this needs a plain HF
  `Trainer` path (or a thin custom loop) rather than `FastModel`. Decide
  whether to generalise `finetune.py` or add a sibling `finetune_hf.py`.
- **Vision-tower SFT**: both expose the vision encoder for training. Mirror
  the Gemma policy knobs (`tune_projector`, `tune_last_n_vision_layers`) so the
  freeze/unfreeze story stays comparable across backbones.
- **Data**: reuse the existing `messages`-format JSONL via `src/data.py`;
  InternVL's chat template and image-tiling preprocessing differ from Qwen's,
  so the collator needs a per-family branch.
- **Policy**: bf16 LoRA only by default (no 4-bit/8-bit) to keep bake-offs
  comparable, same as the Gemma recipe.

This section is a plan, not yet code — it gets its own brainstorm + spec when
prioritised after the ViT baseline lands its first numbers.
