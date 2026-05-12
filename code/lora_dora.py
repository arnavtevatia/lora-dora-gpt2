"""
CS 4782 Project: LoRA + DoRA from Scratch
Arnav Tevatia (at846)

Replicates:
  - LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022)
  - DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024) [extension/ablation]

Model:  GPT-2 small (117M params)
Data:   WikiText-2
Metric: Validation perplexity
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. LoRALinear
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen pretrained nn.Linear with a trainable low-rank update.

    Forward pass:
        h = W0 * x + (alpha / r) * B * A * x

    Initialization:
        A ~ N(0, 1)   (random Gaussian)
        B  = 0        (so LoRA term is zero at init)
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        assert r > 0, "LoRA rank must be positive"

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the original weight
        self.weight = linear.weight  # shape: (out_features, in_features)
        self.bias   = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        in_features  = linear.in_features
        out_features = linear.out_features

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Init: A ~ N(0,1), B = 0
        nn.init.normal_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T  # (..., out_features)
        return base_out + self.scaling * lora_out


# ─────────────────────────────────────────────
# 2. DoRALinear
# ─────────────────────────────────────────────

class DoRALinear(nn.Module):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).

    Decomposes pretrained weight W0 into magnitude m and direction V:
        W0 = m * (V / ||V||_col)

    Applies LoRA to the directional component only:
        W' = m * ((W0 + BA) / ||(W0 + BA)||_col)

    Forward pass:
        h = W' * x + bias

    Key difference from LoRA:
        - Magnitude (m) and direction (W0+BA) are updated separately
        - m is an independent learnable parameter (one scalar per output feature)
        - Leads to more stable training and better performance at same rank
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        assert r > 0

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze original weight
        self.weight = linear.weight  # (out_features, in_features)
        self.bias   = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        in_features  = linear.in_features
        out_features = linear.out_features

        # Trainable LoRA matrices (same as LoRA)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.normal_(self.lora_A)

        # Magnitude vector: initialized to column norms of W0
        # Shape: (out_features, 1) for broadcasting
        with torch.no_grad():
            col_norms = linear.weight.norm(p=2, dim=1, keepdim=True)  # (out_features, 1)
        self.magnitude = nn.Parameter(col_norms.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute adapted weight: W0 + scaling * B * A
        adapted_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)

        # Detach col_norms from gradient graph per DoRA paper Section 4.3.
        # Treating ||V + ΔV||_c as a constant reduces training memory ~24%
        # with negligible accuracy impact. All DoRA paper experiments use this.
        col_norms = adapted_weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).detach()
        directional = adapted_weight / col_norms  # unit direction per output

        # Scale by learned magnitude
        dora_weight = self.magnitude * directional  # (out_features, in_features)

        return nn.functional.linear(x, dora_weight, self.bias)


# ─────────────────────────────────────────────
# 3. GPT-2 Injection
# ─────────────────────────────────────────────

class GPT2AttentionWithLoRA(nn.Module):
    """
    Wraps GPT-2's fused QKV projection (c_attn: Conv1D) to inject
    LoRA or DoRA updates on Q and V only, leaving K unchanged.

    GPT-2 uses Conv1D (not nn.Linear), where weight shape is (in, out).
    We handle this by transposing appropriately.
    """

    def __init__(self, original_attn, r: int, alpha: float, mode: str = "lora"):
        """
        Args:
            original_attn: the GPT2Attention block
            r:             LoRA/DoRA rank
            alpha:         scaling factor
            mode:          "lora" or "dora"
        """
        super().__init__()
        assert mode in ("lora", "dora")
        self.original_attn = original_attn
        self.mode = mode

        # GPT-2 c_attn: Conv1D with weight shape (embed_dim, 3 * embed_dim)
        c_attn = original_attn.c_attn
        in_features  = c_attn.weight.shape[0]   # embed_dim
        out_features = c_attn.weight.shape[1]    # 3 * embed_dim
        head_dim     = out_features // 3         # embed_dim (Q, K, V each)

        AdapterClass = LoRALinear if mode == "lora" else DoRALinear

        # Build a dummy nn.Linear to initialize adapters for Q and V
        # We pass a proxy linear that holds the Q slice and V slice of c_attn
        self.r     = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.head_dim = head_dim
        self.in_features = in_features

        # Low-rank matrices for Q and V
        if mode == "lora":
            # Q adapter
            self.lora_A_q = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_q = nn.Parameter(torch.zeros(head_dim, r))
            nn.init.normal_(self.lora_A_q)
            # V adapter
            self.lora_A_v = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_v = nn.Parameter(torch.zeros(head_dim, r))
            nn.init.normal_(self.lora_A_v)
        else:
            # DoRA: same A, B matrices + magnitude vectors
            self.lora_A_q = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_q = nn.Parameter(torch.zeros(head_dim, r))
            nn.init.normal_(self.lora_A_q)

            self.lora_A_v = nn.Parameter(torch.empty(r, in_features))
            self.lora_B_v = nn.Parameter(torch.zeros(head_dim, r))
            nn.init.normal_(self.lora_A_v)

            # Extract Q and V slices of c_attn weight for magnitude init
            # c_attn.weight shape: (in_features, 3 * head_dim)
            W = c_attn.weight.data  # (in_features, 3*head_dim)
            W_q = W[:, :head_dim].T                    # (head_dim, in_features)
            W_v = W[:, 2*head_dim:3*head_dim].T        # (head_dim, in_features)

            mag_q = W_q.norm(p=2, dim=1, keepdim=True)  # (head_dim, 1)
            mag_v = W_v.norm(p=2, dim=1, keepdim=True)

            self.magnitude_q = nn.Parameter(mag_q.clone())
            self.magnitude_v = nn.Parameter(mag_v.clone())

        # Freeze c_attn weights
        original_attn.c_attn.weight.requires_grad_(False)
        if original_attn.c_attn.bias is not None:
            original_attn.c_attn.bias.requires_grad_(False)

    def forward(self, hidden_states, **kwargs):
        """
        Intercepts the attention forward pass to inject adapter deltas
        into Q and V before calling the rest of the attention block.
        """
        # Get base QKV from frozen c_attn
        # GPT-2 Conv1D: output = x @ weight + bias  (weight is (in, out))
        c_attn = self.original_attn.c_attn
        base_qkv = hidden_states @ c_attn.weight  # (B, T, 3*head_dim)
        if c_attn.bias is not None:
            base_qkv = base_qkv + c_attn.bias

        h = self.head_dim
        scaling = self.scaling

        # Compute LoRA/DoRA deltas for Q
        delta_q = (hidden_states @ self.lora_A_q.T) @ self.lora_B_q.T  # (B,T,head_dim)
        delta_v = (hidden_states @ self.lora_A_v.T) @ self.lora_B_v.T

        if self.mode == "lora":
            base_qkv[:, :, :h]      = base_qkv[:, :, :h]      + scaling * delta_q
            base_qkv[:, :, 2*h:3*h] = base_qkv[:, :, 2*h:3*h] + scaling * delta_v
        else:
            # DoRA: normalize adapted weight direction, scale by magnitude
            W = c_attn.weight.data  # (in_features, 3*head_dim)

            # Q slice
            # Detach norms per DoRA paper Section 4.3 — treats ||V+ΔV||_c
            # as constant during backprop, reducing memory ~24% with negligible accuracy loss.
            W_q = W[:, :h].T  # (head_dim, in_features)
            adapted_q = W_q + scaling * (self.lora_B_q @ self.lora_A_q)
            norm_q = adapted_q.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).detach()
            dora_q = self.magnitude_q * (adapted_q / norm_q)  # (head_dim, in_features)
            # Recompute Q output with DoRA weight
            q_out = hidden_states @ dora_q.T  # (B, T, head_dim)
            if c_attn.bias is not None:
                q_out = q_out + c_attn.bias[:h]
            base_qkv[:, :, :h] = q_out

            # V slice
            W_v = W[:, 2*h:3*h].T  # (head_dim, in_features)
            adapted_v = W_v + scaling * (self.lora_B_v @ self.lora_A_v)
            norm_v = adapted_v.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).detach()
            dora_v = self.magnitude_v * (adapted_v / norm_v)
            v_out = hidden_states @ dora_v.T
            if c_attn.bias is not None:
                v_out = v_out + c_attn.bias[2*h:3*h]
            base_qkv[:, :, 2*h:3*h] = v_out

        # Monkey-patch c_attn to return precomputed QKV
        # We temporarily override to pass through our modified QKV
        original_c_attn = self.original_attn.c_attn

        class _PassthroughConv1D(nn.Module):
            def __init__(self, precomputed):
                super().__init__()
                self.precomputed = precomputed
            def forward(self, x):
                return self.precomputed

        self.original_attn.c_attn = _PassthroughConv1D(base_qkv)
        out = self.original_attn(hidden_states, **kwargs)
        self.original_attn.c_attn = original_c_attn
        return out


# ─────────────────────────────────────────────
# 4. Model Setup
# ─────────────────────────────────────────────

def inject_adapters(model: GPT2LMHeadModel, r: int, alpha: float, mode: str) -> GPT2LMHeadModel:
    """
    Injects LoRA or DoRA into all GPT-2 attention blocks (Q and V only).
    Freezes all other parameters.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Inject adapters into each attention layer
    for block in model.transformer.h:
        block.attn = GPT2AttentionWithLoRA(block.attn, r=r, alpha=alpha, mode=mode)

    return model


def count_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable
    print(f"  Trainable params : {trainable:>12,}  ({100*trainable/total:.4f}%)")
    print(f"  Frozen params    : {frozen:>12,}")
    print(f"  Total params     : {total:>12,}")
    return trainable


# ─────────────────────────────────────────────
# 5. Data
# ─────────────────────────────────────────────

def get_dataloaders(tokenizer, block_size: int = 512, batch_size: int = 4):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated["input_ids"])
        total = (total // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    lm_dataset = tokenized.map(group_texts, batched=True)
    lm_dataset.set_format(type="torch")

    train_loader = DataLoader(lm_dataset["train"],      batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(lm_dataset["validation"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 6. Training & Evaluation
# ─────────────────────────────────────────────

def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            outputs   = model(input_ids=input_ids, labels=labels)
            loss      = outputs.loss
            num_tokens = (labels != -100).sum().item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def train_one_epoch(model, train_loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, steps = 0.0, 0
    for batch in tqdm(train_loader, desc="  Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        outputs   = model(input_ids=input_ids, labels=labels)
        loss      = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total_loss / steps


def run_experiment(
    config_name: str,
    mode: str,           # "frozen", "full", "lora", "dora"
    r: int = 4,
    alpha: float = 1.0,
    lr: float = 3e-4,
    epochs: int = 3,
    batch_size: int = 4,
    block_size: int = 512,
    device: str = "cuda",
    max_train_steps: int = None,   # set small (e.g. 50) for quick smoke test
):
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  |  mode={mode}  r={r}  alpha={alpha}")
    print(f"{'='*60}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = get_dataloaders(tokenizer, block_size, batch_size)

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)

    elif mode in ("lora", "dora"):
        model = inject_adapters(model, r=r, alpha=alpha, mode=mode)

    model = model.to(device)

    print("\nParameter audit:")
    trainable = count_parameters(model)

    # Peak GPU memory after first forward pass
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Baseline perplexity (before any training)
    ppl_before = evaluate(model, val_loader, device)
    print(f"\n  Perplexity before training: {ppl_before:.2f}")

    if mode == "frozen" or trainable == 0:
        print("  (No trainable params, skipping training)")
        return {"config": config_name, "mode": mode, "r": r,
                "trainable_params": trainable, "perplexity": ppl_before,
                "peak_gpu_mb": None, "time_per_epoch_s": None}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    results = {"config": config_name, "mode": mode, "r": r, "trainable_params": trainable}
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, max_steps=max_train_steps)
        elapsed  = time.time() - t0
        epoch_times.append(elapsed)

        ppl = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_ppl={ppl:.2f}  "
              f"time={elapsed:.1f}s")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_mb = None

    results["perplexity"]        = ppl
    results["peak_gpu_mb"]       = peak_mb
    results["time_per_epoch_s"]  = sum(epoch_times) / len(epoch_times)

    print(f"\n  Final val perplexity : {ppl:.2f}")
    if peak_mb:
        print(f"  Peak GPU memory      : {peak_mb:.1f} MB")
    print(f"  Avg time/epoch       : {results['time_per_epoch_s']:.1f}s")

    return results


# ─────────────────────────────────────────────
# 7. Main: run all 6 configurations
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # Set MAX_TRAIN_STEPS=None for full training.
    # Set to a small int (e.g. 20) for a quick smoke test to check
    # everything runs without errors before committing GPU time.
    MAX_TRAIN_STEPS = None  # change to 20 for smoke test

    configs = [
        # (config_name,          mode,   r,    alpha)
        ("Frozen GPT-2",         "frozen", None, None),
        ("Full Fine-tuning",     "full",   None, None),
        ("LoRA r=4",             "lora",   4,    1.0),
        ("LoRA r=8",             "lora",   8,    1.0),
        ("DoRA r=4",             "dora",   4,    1.0),
        ("DoRA r=8",             "dora",   8,    1.0),
    ]

    all_results = []
    for name, mode, r, alpha in configs:
        r_arg     = r     if r     is not None else 4
        alpha_arg = alpha if alpha is not None else 1.0
        res = run_experiment(
            config_name=name,
            mode=mode,
            r=r_arg,
            alpha=alpha_arg,
            epochs=3,
            batch_size=4,
            block_size=512,
            device=DEVICE,
            max_train_steps=MAX_TRAIN_STEPS,
        )
        all_results.append(res)

    # ── Summary Table ──────────────────────────────────────────
    print("\n\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<22} {'Mode':<8} {'r':<5} {'Trainable':<14} "
          f"{'Val PPL':<10} {'Peak GPU (MB)':<15} {'Avg Epoch (s)'}")
    print("-"*80)
    for r in all_results:
        tp  = f"{r['trainable_params']:,}"   if r['trainable_params'] else "0"
        ppl = f"{r['perplexity']:.2f}"       if r['perplexity']       else "N/A"
        mem = f"{r['peak_gpu_mb']:.1f}"      if r['peak_gpu_mb']      else "N/A"
        t   = f"{r['time_per_epoch_s']:.1f}" if r['time_per_epoch_s'] else "N/A"
        rank = str(r['r']) if r['r'] else "-"
        print(f"{r['config']:<22} {r['mode']:<8} {rank:<5} {tp:<14} {ppl:<10} {mem:<15} {t}")
    print("="*80)
