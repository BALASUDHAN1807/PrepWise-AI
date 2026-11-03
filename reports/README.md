# reports/

Output of the experiment scripts lands here (`*.md` is human-readable, `*.json` has raw rows and is git-ignored):

| file | produced by |
|---|---|
| `classifier_eval.md` | `python scripts/train_classifier.py` (deterministic, no API needed) |
| `prompt_comparison.md` | `python scripts/prompt_comparison.py` |
| `model_comparison.md` | `python scripts/model_comparison.py --models "gemini,openai,anthropic,ollama:llama3.2"` |
| `eval_consistency.md` | `python scripts/eval_consistency.py` |
| `compare_approaches.md` | `python scripts/compare_approaches.py` (needs `models/lora_flan_t5_small` from `scripts/finetune_lora.py` for the LoRA row) |

The three LLM experiments are only meaningful with a real provider configured in `.env`. With `LLM_PROVIDER=mock`
every strategy/model returns the same rule-based answer, and each report says so at the top. Commit the reports you
generate with a real provider so reviewers can see your numbers.
