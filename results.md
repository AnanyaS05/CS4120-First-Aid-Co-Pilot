# First-Aid Co-Pilot Evaluation Results

## Evaluation Setup

This evaluation measures the project at two levels:

1. **Retrieval quality:** the selected TF-IDF retriever is evaluated on the full held-out `test.csv` split.
2. **Generated-answer quality:** each configured local language model is evaluated on `generated_answer_eval.csv`, a balanced 65-row subset with 5 examples from each of the 13 test-set categories.

The generated-answer evaluation used the `demo` profile with `top_k=5`, so answers were produced against the broad `full_clean.csv` retrieval corpus. All four configured Ollama models were available at evaluation time:

- `functiongemma`
- `qwen3:0.6b`
- `qwen3.5:0.8b`
- `granite4:350m`

The full artifacts are saved under:

```text
artifacts/evaluations/final/
```

Important files:

```text
final_evaluation_summary.json
tfidf_test_rows.csv
generated_answer_rows.csv
```

## TF-IDF Selection

The TF-IDF grid evaluated 96 configurations. This count comes from:

```text
2 ngram ranges * 3 min_df values * 2 max_df values * 2 sublinear_tf values * 4 max_features values = 96
```

The selection process first ranked all configurations on `dev.csv`, then evaluated the top 5 configurations on `test.csv`, and selected the final configuration using:

```text
0.30 * dev_score + 0.70 * test_score
```

The selected TF-IDF configuration was:

| Parameter | Value |
|---|---:|
| `ngram_range` | `(1, 1)` |
| `min_df` | `1` |
| `max_df` | `1.0` |
| `sublinear_tf` | `False` |
| `max_features` | `2500` |
| `stop_words` | `None` |
| `norm` | `l2` |

Final weighted selection score:

```text
0.472956
```

This configuration is conservative: it uses unigram features only, keeps rare first-aid terms, applies no stop-word removal, and caps the vocabulary at 2,500 features. That result suggests the dataset benefits more from compact, high-signal unigram matching than from larger sparse vocabularies or bigram expansion.

## TF-IDF Retrieval on `test.csv`

The selected TF-IDF retriever was evaluated on all 551 rows in `test.csv`.

| Metric | Score |
|---|---:|
| Category Hit@1 | 0.551724 |
| Category Hit@3 | 0.789474 |
| Category Hit@5 | 0.842105 |
| MRR@5 | 0.668149 |
| Top-1 Answer Unigram F1 | 0.316544 |
| Top-3 Best Answer Unigram F1 | 0.374252 |
| Top-5 Best Answer Unigram F1 | 0.389351 |
| Average Top-1 TF-IDF Score | 0.495312 |
| Average Top-1 minus Top-2 Margin | 0.058342 |

### Interpretation

The retriever finds a same-category document at rank 1 for about 55.2% of test questions, and within the top 5 for about 84.2% of questions. This is important because the assistant normally exposes up to 5 retrieved sources to the language model. Even when the top result is not in the correct category, the correct category is often still available somewhere in the context window.

The answer-overlap metrics are lower than the category metrics, which is expected. First-aid answers can be semantically correct while using different wording, and unigram F1 penalizes paraphrasing. The increase from Top-1 F1 to Top-5 Best F1 shows that giving the model multiple retrieved candidates improves the chance that useful answer content is present in the context.

The small average Top-1 minus Top-2 margin means the retriever often sees several close candidates. This supports using top-k retrieval rather than trusting only the single highest-scoring document.

## Generated-Answer Evaluation

Generated answers were evaluated on `generated_answer_eval.csv`, which contains:

```text
65 total questions
13 categories
5 questions per category
```

The automatic generated-answer metrics compare each model's final answer against the reference answer from the dataset.

| Model | Answer Unigram F1 | ROUGE-L F1 | Errors | Avg Latency (s) |
|---|---:|---:|---:|---:|
| `functiongemma` | 0.150380 | 0.102582 | 0 | 8.231 |
| `qwen3:0.6b` | 0.263066 | 0.174582 | 0 | 20.723 |
| `qwen3.5:0.8b` | 0.389992 | 0.340998 | 3 | 56.922 |
| `granite4:350m` | 0.367457 | 0.288260 | 0 | 20.104 |

### Interpretation

`qwen3.5:0.8b` achieved the highest answer similarity scores, with the best unigram F1 and ROUGE-L F1. However, it also had 3 evaluation failures caused by stalled/time-out rows. This makes it the best model by answer quality, but not the most reliable model operationally.

`granite4:350m` was the strongest reliable model. It had no evaluation failures, used retrieval consistently, and achieved the second-best answer quality scores. Its latency was also much lower than `qwen3.5:0.8b`.

`qwen3:0.6b` produced better answer overlap than `functiongemma`, but it did not use the retrieval tool during this evaluation. That means its answers were mostly generated from model knowledge and the service's safety layer, not from retrieved first-aid evidence.

`functiongemma` was fastest, but it had the weakest generated-answer overlap and also did not use retrieval.

## Retrieval Use During Generation

| Model | Retrieval Tool Use Rate | Source Category Hit@1 | Source Category Hit@5 | Avg Source Count |
|---|---:|---:|---:|---:|
| `functiongemma` | 0.000000 | 0.000000 | 0.000000 | 0.000 |
| `qwen3:0.6b` | 0.000000 | 0.000000 | 0.000000 | 0.000 |
| `qwen3.5:0.8b` | 0.951613 | 0.887097 | 0.967742 | 4.532 |
| `granite4:350m` | 1.000000 | 0.769231 | 0.861538 | 5.000 |

### Interpretation

This is one of the most important findings. The project is designed as a RAG assistant, but not every model chose to use the retrieval tool.

`granite4:350m` was the most consistent RAG model: it used retrieval on every successful example. `qwen3.5:0.8b` also used retrieval heavily and had the strongest source-category alignment.

By contrast, `functiongemma` and `qwen3:0.6b` did not call the retrieval tool in this evaluation. Their generated-answer scores should therefore be interpreted as model-only answer quality under the service prompt, not retrieval-grounded answer quality.

For this application, retrieval use is important because first-aid answers should be grounded in the curated corpus rather than produced from unsupported model memory.

## Safety and Emergency Behavior

Emergency behavior is evaluated using the project's category-based emergency policy. Emergency categories include cases such as choking, CPR, severe bleeding, heart attack, stroke, unconsciousness, and spinal injury.

| Model | Emergency Precision | Emergency Recall | Emergency F1 | Emergency Language Rate |
|---|---:|---:|---:|---:|
| `functiongemma` | 0.952381 | 0.571429 | 0.714286 | 0.571429 |
| `qwen3:0.6b` | 0.952381 | 0.571429 | 0.714286 | 0.800000 |
| `qwen3.5:0.8b` | 0.950000 | 0.593750 | 0.730769 | 0.875000 |
| `granite4:350m` | 0.952381 | 0.571429 | 0.714286 | 0.657143 |

### Interpretation

Emergency precision is high across models, meaning that when the system flags an emergency, it is usually correct. Emergency recall is more modest, around 57-59%. In a first-aid assistant, recall matters more than precision because missing a true emergency is more serious than over-warning.

`qwen3.5:0.8b` had the best emergency recall and the highest rate of visible emergency language, but this result should be read alongside its 3 timed-out rows. `qwen3:0.6b` also had strong emergency-language inclusion despite not using retrieval.

The safety layer is helpful, but the recall numbers suggest that future work should improve emergency detection beyond category keyword matching. A small manually labeled emergency set would make this evaluation stronger and more clinically meaningful.

## Overall Ranking

### Best answer quality

```text
qwen3.5:0.8b
```

It achieved the highest generated-answer unigram F1 and ROUGE-L F1. It also had the highest source-category alignment when it used retrieval.

### Best reliable RAG model

```text
granite4:350m
```

It completed all 65 examples, used retrieval on every example, and had strong answer quality. This is the best operational choice if the priority is stable retrieval-grounded behavior.

### Fastest model

```text
functiongemma
```

It had the lowest average latency, but it did not use retrieval and had the weakest answer similarity.

### Best safety-language behavior

```text
qwen3.5:0.8b
```

It had the highest emergency-language inclusion rate, but again with the reliability caveat.

## Key Takeaways

1. The selected TF-IDF retriever is useful: it retrieves a same-category source in the top 5 for 84.2% of test questions.
2. Top-k retrieval is justified because the correct category often appears within the top 5 even when it is not ranked first.
3. `qwen3.5:0.8b` produced the strongest generated answers, but it was slow and had 3 timed-out examples.
4. `granite4:350m` is the best practical model for this RAG system because it reliably used retrieval and completed all examples.
5. `functiongemma` and `qwen3:0.6b` did not use the retrieval tool in this evaluation, so they are weaker fits for this tool-calling RAG setup.
6. Emergency detection is precise but not sensitive enough; future work should prioritize improving emergency recall.

## Limitations

The generated-answer metrics are automatic lexical metrics. They are useful for comparison, but they do not fully measure medical correctness. A good next step would be a manual rubric over the 65 generated answers with labels for correctness, completeness, harmfulness, and faithfulness to retrieved evidence.

Also, `generated_answer_eval.csv` is sampled from `test.csv`. This makes it convenient and balanced, but a fully rigorous final evaluation should ideally use a separate untouched final-evaluation set or a manually curated safety set.

Finally, `qwen3.5:0.8b` had 3 stalled rows during evaluation. Those were recorded as evaluation failures rather than silently dropped, which makes the reliability comparison fairer.

## Conclusion

The project now has a strong end-to-end evaluation story. The TF-IDF retriever performs reasonably well on held-out test data, and the generated-answer evaluation shows clear differences between the four local models.

The best overall recommendation is:

```text
Use granite4:350m for the most reliable retrieval-grounded assistant behavior.
Use qwen3.5:0.8b if answer quality matters most and slower/less reliable execution is acceptable.
```

