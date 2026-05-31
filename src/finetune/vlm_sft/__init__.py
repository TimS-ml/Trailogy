"""HF-native vision-tower SFT for non-Gemma VLM backbones (Qwen3.5, InternVL3.5).

The shipped Gemma recipe uses unsloth `FastModel`, which only supports the Gemma
family. Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`) and InternVL3.5-4B
(`InternVLForConditionalGeneration`, the `-HF` checkpoint) are both native in
transformers 5.x, so this package drives them through the uniform
`AutoModelForImageTextToText` + `AutoProcessor` path with a plain HF `Trainer`.

It reuses the existing config schema (`src.config`) and JSONL/messages format
(`src.data.build_vision_messages`), so the configs live alongside the Gemma
sweeps under `configs/local_sweep/` and only the backbone + tuning mode change.

Three tuning modes, expressed with existing LoRA-config flags:

    finetune_vision_layers  finetune_language_layers  meaning
    ----------------------  ------------------------  ----------------------
    true                    false                     vision tower only
    true                    true                      vision tower + LLM LoRA r
    false                   true                      LLM LoRA r only
"""
