# LoRA + DoRA as Extension: Weight-Decomposed Low-Rank Adaptation on GPT-2

**CS 4782 Deep Learning - Cornell University - Spring 2026**
Arnav Tevatia (at846)

---

## Introduction

This repo contains a from-scratch PyTorch implementation of **LoRA** (Hu et al., ICLR 2022) applied to GPT-2 small, with **DoRA** (Liu et al., ICML 2024) added as my own extension. No PEFT library is used - all adapter modules are implemented directly.

LoRA freezes pretrained weights and injects trainable low-rank matrix pairs BA to approximate weight updates, reducing trainable parameters from 124M to ~147K at r=4. DoRA extends this by decomposing weights into magnitude and direction components, applying LoRA only to the directional part for more stable gradient dynamics. I was particularly interested in whether DoRA's advantage holds at smaller model scales, since the original paper's results are on LLaMA-7B and larger.

---

## Chosen Result

I replicate LoRA's core claim from **Table 11** of the paper: LoRA matches or exceeds full fine-tuning perplexity on GPT-2 with far fewer trainable parameters. I also replicate the rank sensitivity finding from **Table 18**, which shows diminishing returns past r=4.

As my extension, I add DoRA at matching rank configurations (r=4, r=8) to test whether the magnitude-direction decomposition improves over LoRA at the same parameter budget.

---

## Repo Structure

```
README.md
LICENSE
.gitignore
code/           # Implementation (lora_dora.py)
data/           # Instructions for obtaining WikiText-2
results/        # Training logs and result figures
poster/         # PDF of in-class poster presentation
report/         # PDF of 2-page project report
```

---

## Re-implementation Details

**Model:** GPT-2 small (117M parameters), loaded from HuggingFace pretrained checkpoint.

**Dataset:** WikiText-2 (`wikitext-2-raw-v1`), tokenized into 512-token blocks, causal language modeling objective. Evaluated by validation perplexity.

**Key classes in `code/lora_dora.py`:**
- `LoRALinear` - wraps a frozen `nn.Linear`, adds trainable A (r x d, Gaussian init) and B (d x r, zeros init). Forward: W0x + (a/r) * (xA^T)B^T
- `DoRALinear` - extends LoRALinear with a magnitude vector m. Forward: m * (W0 + BA) / ||W0 + BA||_c. Column norms are detached from the gradient graph (DoRA Section 4.3) to reduce training memory ~24% with <0.2% accuracy impact.
- `GPT2AttentionWithLoRA` - intercepts GPT-2's fused `c_attn` projection and applies LoRA/DoRA deltas to Q and V slices only (K unchanged, per LoRA Table 5).

**Training:** AdamW, gradient clipping at max_norm=1.0, 3 epochs, lr=3e-4, batch size 4.

**Verification steps before training:**
1. Parameter audit - only A, B, and magnitude vectors have `requires_grad=True`
2. Numerical equivalence check - model output identical to pretrained GPT-2 at initialization (B=zeros ensures zero LoRA term)

---

## Reproduction Steps

**Requirements:**
```
torch
transformers
datasets
```

Install with:
```bash
pip install torch transformers datasets
```

**Run all 6 configurations:**
```bash
python code/lora_dora.py
```

This runs sequentially: Frozen GPT-2, Full fine-tuning, LoRA r=4, LoRA r=8, DoRA r=4, DoRA r=8. Results are printed to stdout at the end.

**GPU requirements:** Single GPU with at least 8GB VRAM (tested on T4). Estimated total runtime: 10-17 hours for all 6 configs.

---

## Results

| Configuration | Trainable Params | Val PPL | Peak GPU (MB) |
|---|---|---|---|
| Frozen GPT-2 | 0 | 36.08 | N/A |
| Full fine-tune | 124,440K | 32.13 | 5,753 |
| LoRA r=4 | 147K | 25.65 | 4,118 |
| LoRA r=8 | 295K | 25.26 | 4,120 |
| DoRA r=4 | 166K | 25.50 | 4,227 |
| DoRA r=8 | 313K | **25.22** | 4,228 |

LoRA beats full fine-tuning despite using 843x fewer parameters. I did not expect this going in - it turns out the low-rank constraint acts as implicit regularization on WikiText-2's small training set, and full fine-tuning actually overfits after epoch 1. DoRA consistently outperforms LoRA at the same rank, though the gap is smaller than the paper's LLaMA-7B results, which makes sense given the simpler task and smaller model.

---

## Conclusion

LoRA's value on small datasets goes beyond parameter efficiency - the low-rank constraint also leads to better generalization by preventing overfitting. DoRA provides a consistent small improvement over LoRA with negligible extra cost. Whether that advantage grows with model scale is an open question and a natural next step.

---

## References

1. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685
2. Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024. arXiv:2402.09353
3. Radford et al. GPT-2. OpenAI, 2019.
4. Merity et al. WikiText-2. arXiv:1609.07843
5. Wolf et al. HuggingFace Transformers. EMNLP 2020.

---

## Acknowledgements

This project was completed as the CS 4782 (Deep Learning) final project at Cornell University, Spring 2026, under Professor Kilian Weinberger. All implementation is from scratch without any PEFT library.