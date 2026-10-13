# Evaluation notes (draft for TRAINING_RUN.md)

## Setup

- Model: `small` config of `model.py` (GQA 6:2 heads, SwiGLU, RMSNorm, RoPE, 32k vocab), 34,018,176 parameters.
- Data: `data/re-corpus.jsonl`, split by document: 618 train docs (1,665,029 tokens), 32 val docs (161,689 tokens).
- Training: 2,400 steps, batch 16 x 256 tokens, AdamW, cosine LR 1e-3 to 1e-4 with 100 warmup steps, CPU. That is about 5.9 passes over the train tokens. Wall time was 4,682 s.
- Evaluation (`eval.py`): mean next-token cross-entropy over the **full** val split, using contiguous non-overlapping 256-token windows. This scores 161,536 target tokens. Perplexity = exp(loss).
- Baseline: a unigram model built from train-split token counts with add-one smoothing over the 32,000-token vocabulary. It is scored on the same 161,536 val targets. 1.85% of those val targets never occur in the train split.

## Results (full val split)

| checkpoint | step | train loss (log) | val loss | val ppl |
|---|---|---|---|---|
| unigram baseline | – | – | 7.7180 | 2248.43 |
| ckpt-0000600 | 600 | 3.7715 | 5.2393 | 188.54 |
| ckpt-0001200 | 1200 | 2.9800 | 4.7767 | 118.72 |
| ckpt-0001800 | 1800 | 1.9978 | 4.7527 | 115.89 |
| best.pt | 2000 | 2.0727 | **4.7251** | **112.74** |
| final.pt | 2400 | 1.7107 | 4.8019 | 121.74 |

"Train loss (log)" is the mean training loss over the 50 steps before each checkpoint, as written in `log.csv`.

## Observations

- On the full val split, the best checkpoint's loss is 2.99 nats/token lower than the unigram baseline (perplexity 112.7 vs 2248.4).
- **Overfitting:** after step 1200, val loss levels off and then rises (4.7251 at step 2000, 4.8019 at step 2400). Over the same steps, train loss keeps falling, to 1.71 at step 2400. At the final step the gap between train and full-val loss is 3.09 nats. With about 1.7M train tokens and 34M parameters, the model memorizes the corpus. Use `best.pt` (step 2000), not `final.pt`.
- **Sampled vs full val:** during training, `train.py` scores val on only the first 320 windows (`--eval-batches 20`), which is about half the val split. That gives lower numbers (3.7010 at step 2000, 3.7609 at step 2400) than the full split (4.7251, 4.8019). The second half of the val documents is harder for the model. Compare only full-split numbers, which `report.json` records as `final_val_loss_full`, against each other and against the baseline.
- **Samples:** greedy decoding and T=0.8 decoding (48 new tokens, prompts from `data/re-prompts.txt`) produce text with the right local form: Markdown headings, bullet lists, inline code, and terms from the corpus. The text does not answer the prompts and is not factually coherent, and greedy decoding often repeats itself. This is the expected behavior for a 34M-parameter model trained on 1.7M tokens.
- **Plot:** `docs/train-loss-v1.png` shows train loss, sampled val loss, and the unigram baseline.

## Reproduce

    python3 pytorch-4b/eval.py --ckpt runs/small-cpu/best.pt --threads 2 --json eval-best.json
    python3 pytorch-4b/plot_loss.py --log runs/small-cpu/log.csv --unigram 7.718
