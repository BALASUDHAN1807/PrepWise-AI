# Soft-skill classifier evaluation

| variant | random CV acc (mean±sd) | leave-questions-out acc | macro-F1 (LQO) |
|---|---|---|---|
| length_only | 0.6845 ± 0.0427 | 0.6814 | 0.6691 |
| tfidf_only | 0.7355 ± 0.0536 | 0.7794 | 0.7762 |
| tfidf+features | 0.9087 ± 0.0498 | 0.8922 | 0.893 |

Hold-out (selected model): {'accuracy': 0.8824, 'precision_macro': 0.8832, 'recall_macro': 0.8852, 'f1_macro': 0.8836, 'confusion_matrix': [[16, 2, 0], [1, 15, 2], [0, 1, 14]], 'labels': ['poor', 'average', 'good']}

> Small LLM-authored dataset in which length and structure strongly track the label (see the length_only baseline). Accuracy here does NOT show real-world interview-grading skill.
