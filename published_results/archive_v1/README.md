# Archive — ChaosBench-Logic v1 Results

> **⚠️ ARCHIVED — NOT COMPARABLE TO v2**
>
> These results were produced on the **v1 dataset** (621 questions, pre-2026 schema).
> The v2 dataset has 40,886 questions with a completely different family structure and
> stricter ground-truth verification. v1 and v2 metrics **MUST NOT** appear in the same table.

## Contents

| Run | Model | Mode | N | Date |
|-----|-------|------|---|------|
| `claude3_zeroshot/` | Claude-3.5-Sonnet | zero-shot | 620 | 2025-12-14 |
| `gemini_zeroshot/` | Gemini-2.0-Flash | zero-shot | 620 | 2025-12-14 |
| `gpt4_cot/` | GPT-4-Turbo | chain-of-thought | 620 | 2025-12-14 |
| `gpt4_zeroshot/` | GPT-4-Turbo | zero-shot | 620 | 2025-12-14 |
| `llama3_cot/` | LLaMA-3-70B | chain-of-thought | 620 | 2025-12-14 |
| `llama3_zeroshot/` | LLaMA-3-70B | zero-shot | 620 | 2025-12-14 |

## Why These Are Archived

1. **Dataset change**: v2 extends from 621 → 40,886 questions; families are renamed and restructured.
2. **Schema change**: v1 `summary.json` uses `overall_accuracy` / `task_accuracy` keys;
   v2 uses `balanced_accuracy` / `mcc` / `per_family`.
3. **Ground-truth revision**: v1 ground truth was not freeze-verified; v2 uses a cryptographically
   frozen dataset with per-file SHA256 checksums.
4. **Parsing**: v1 runs did not use strict parsing; v2 requires `strict_parsing=true`.

## Historical Reference

These results are retained for completeness. They are not used in any v2 paper tables or figures.
For v2 results, see `../runs/README.md`.
