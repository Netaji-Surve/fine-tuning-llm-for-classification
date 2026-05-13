"""
config.py — All hyperparameters for the QLoRA experiment, in one place.

WHY a separate config file?
  When you run experiments, you tweak numbers constantly: change the rank,
  try a different learning rate, use more data. If those numbers are scattered
  across 5 files, one change means hunting through all of them. Putting
  every knob in one dataclass means you change ONE line and the rest of the
  code just picks it up automatically.

WHY @dataclass instead of a plain dict?
  A dict like {"r": 16, "alpha": 32} gives you no type hints, no IDE
  autocomplete, and no default values. A @dataclass gives you all three,
  plus you can pass it as a typed argument to functions.
"""

# dataclass: a decorator that auto-generates __init__, __repr__, __eq__
# from the class attributes you declare. Saves a lot of boilerplate.
from dataclasses import dataclass, field

# List: a type hint that says "this attribute holds a Python list"
from typing import List


# ─────────────────────────────────────────────────────────────
#  1.  LoRA adapter configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """
    Controls the LOW-RANK ADAPTATION (LoRA) adapter that is attached to GPT-2.

    ┌─ CONCEPT: What IS LoRA? ──────────────────────────────────────────────────┐
    │                                                                            │
    │  A pre-trained model weight matrix W has shape (d_in × d_out).            │
    │  Full fine-tuning updates every element of W — that is d_in × d_out       │
    │  parameters per layer.                                                     │
    │                                                                            │
    │  LoRA (Hu et al., 2021) says: "we don't need to update W directly.        │
    │  Instead, represent the UPDATE ΔW as a product of two tiny matrices:"     │
    │                                                                            │
    │      ΔW  ≈  B × A                                                         │
    │                                                                            │
    │  where:                                                                    │
    │    A  has shape  (r × d_in)   ← r rows, d_in columns                      │
    │    B  has shape  (d_out × r)  ← d_out rows, r columns                     │
    │    r  is the RANK  (a small number like 8 or 16)                           │
    │                                                                            │
    │  The forward pass becomes:                                                 │
    │      y  =  W₀ x  +  (α/r) · B A x                                         │
    │            ────         ──────────                                         │
    │           frozen         LoRA adapter (trainable)                          │
    │                                                                            │
    │  Parameters trained:                                                       │
    │    Full fine-tuning: d_in × d_out  (e.g., 768 × 2304 = 1,769,472)        │
    │    LoRA (r=16):      r×d_in + r×d_out = 16×768 + 16×2304 = 49,152        │
    │    → 97% fewer parameters!                                                 │
    └────────────────────────────────────────────────────────────────────────────┘
    """

    r: int = 16
    # The RANK of the LoRA decomposition.
    #
    # Think of rank as the "information bottleneck" between A and B.
    # A rank-r matrix can represent at most r independent directions.
    # The weight update is forced through this bottleneck.
    #
    # Why does this still work? Research shows that during fine-tuning,
    # the meaningful weight changes lie in a LOW-DIMENSIONAL subspace —
    # you don't need to update all d_in × d_out directions, just a few.
    #
    # Practical guide:
    #   r = 4  → fewest params, fastest training, works for simple tasks
    #   r = 8  → good default for most classification tasks
    #   r = 16 → our default: a safe balance of capacity and efficiency
    #   r = 32 → use when r=16 plateaus; doubles the adapter size
    #   r = 64 → rarely needed; approaching full fine-tuning territory

    lora_alpha: int = 32
    # A SCALING FACTOR applied to the LoRA output.
    #
    # The actual adapter contribution is: (lora_alpha / r) × B × A × x
    # With lora_alpha=32, r=16:  scale = 32/16 = 2.0
    #
    # WHY do we need this?
    #   When you change r, the magnitude of B×A changes too (because B and A
    #   are initialized with values scaled by 1/r). lora_alpha lets you
    #   control the effective "strength" of the adapter independently from r.
    #
    # Rule of thumb:
    #   Keep lora_alpha = 2 × r  →  scale stays ~2 regardless of r.
    #   Or: lora_alpha = r  →  scale = 1.0  (neutral, let gradients decide).

    lora_dropout: float = 0.1
    # DROPOUT probability applied to the LoRA adapter output during training.
    #
    # Dropout: at each forward pass, randomly zero out 10% of the adapter's
    # output values. This means the model cannot rely on any single adapter
    # neuron — it must spread knowledge across many neurons.
    #
    # Effect: acts as regularisation, reducing overfitting on small datasets.
    # At inference (model.eval()), dropout is automatically disabled — all
    # values pass through unchanged.
    #
    # 0.0 = no dropout (use this if you have a large dataset)
    # 0.1 = 10% dropout (safe default for medium datasets like ours)
    # 0.2 = 20% dropout (try if model is clearly overfitting)

    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    # WHICH layers inside GPT-2 to attach LoRA adapters to.
    #
    # ┌─ GPT-2 Attention Architecture ────────────────────────────────────────┐
    # │  Input  →  c_attn (768 → 2304)  →  split into Q, K, V                │
    # │                                        ↓                              │
    # │                              Scaled Dot-Product Attention             │
    # │                                        ↓                              │
    # │                         c_proj (768 → 768)  →  Output                │
    # └────────────────────────────────────────────────────────────────────────┘
    #
    # c_attn: a single Conv1D that computes Query, Key, Value all at once.
    #   Input: (batch, seq_len, 768) → Output: (batch, seq_len, 2304)
    #   The 2304 = 3 × 768 is then split into Q (768), K (768), V (768).
    #   By adapting this layer, we change WHAT the model attends to.
    #
    # c_proj: projects the attention output back to hidden dimension.
    #   Input: (batch, seq_len, 768) → Output: (batch, seq_len, 768)
    #   By adapting this layer, we change HOW attention results are processed.
    #
    # Together, these two layers govern all attention patterns in GPT-2.
    # Adapting them lets the model re-learn "what to look for" in spam emails
    # (suspicious phrases, excessive caps, urgency signals) without touching
    # the feed-forward layers that store factual/world knowledge.
    #
    # Alternative: add "c_fc" (the first FFN layer) for more capacity,
    # but attention layers alone usually suffice for classification.

    bias: str = "none"
    # Whether to also train BIAS vectors alongside the A and B matrices.
    #
    # Every linear layer has the form: y = Wx + b, where b is the bias.
    # By default, we freeze all biases ("none") for maximum efficiency.
    #
    # Options:
    #   "none"      → freeze all biases. Recommended — saves params, works well.
    #   "all"       → train every bias in the entire model. Many more params.
    #   "lora_only" → train only the biases in LoRA-adapted layers.


# ─────────────────────────────────────────────────────────────
#  2.  Training loop configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Controls the training loop: how long to train, how fast to learn,
    when to save, what to optimise for.

    These are passed directly to HuggingFace's TrainingArguments, which
    drives the Trainer class under the hood.
    """

    output_dir: str = "src/qlora/results"
    # Folder where training checkpoints and the final model are saved.
    # HuggingFace Trainer creates this directory if it doesn't exist.
    # A "checkpoint" is a snapshot of the model weights at a specific step,
    # so you can resume training or roll back if something goes wrong.

    num_train_epochs: int = 3
    # How many times the model sees the ENTIRE training dataset.
    # 1 epoch = one complete pass over every training example.
    #
    # After each epoch:
    #   - We evaluate on the validation set (because evaluation_strategy="epoch")
    #   - We save a checkpoint (because save_strategy="epoch")
    #
    # 3 epochs is a common starting point:
    #   - Too few epochs → model under-trains (doesn't learn the task well)
    #   - Too many epochs → model over-trains (memorises training data, fails on new data)

    per_device_train_batch_size: int = 8
    # How many examples to process in ONE forward + backward pass.
    #
    # "per_device" means PER GPU/MPS device. If you have 2 GPUs, the actual
    # batch size is 2 × 8 = 16 (Trainer handles the split automatically).
    #
    # Trade-offs:
    #   Large batch (32, 64):  stable gradient estimates, but needs more memory
    #   Small batch (4, 8):    noisier gradients, but uses less memory and can
    #                          act as implicit regularisation
    #
    # 8 is safe for GPT-2 on 16 GB Apple Silicon or any modern GPU.

    per_device_eval_batch_size: int = 16
    # Batch size used during EVALUATION (validation set only).
    # No backward pass happens during eval → no gradient memory needed.
    # So we can use a larger batch for faster evaluation.

    learning_rate: float = 2e-4
    # How large each optimiser step is. (2e-4 = 0.0002)
    #
    # ┌─ WHY 2e-4 instead of 2e-5 used in the baseline GPT-2 notebook? ──────┐
    # │                                                                        │
    # │  In full fine-tuning, you update 124M parameters that already contain  │
    # │  rich pre-trained knowledge. A large learning rate can DESTROY that    │
    # │  knowledge (catastrophic forgetting). So you use a small LR (2e-5).   │
    # │                                                                        │
    # │  With LoRA, only the tiny A and B matrices are trained (~0.5% of      │
    # │  params). The frozen backbone is never touched, so there is no risk    │
    # │  of catastrophic forgetting. A larger LR makes the adapters converge  │
    # │  faster without any risk. 2e-4 is the community standard for LoRA.    │
    # └────────────────────────────────────────────────────────────────────────┘

    warmup_steps: int = 100
    # For the first 100 training steps, the LR linearly increases from 0 → learning_rate.
    #
    # WHY? At the very start of training, the LoRA weights A and B are random
    # (A is random, B is zero). Gradients in the first few steps are very noisy.
    # A full-size learning rate on noisy gradients → wild, damaging updates.
    # Warmup gently eases in, giving the optimiser time to "calibrate".
    #
    # After warmup, the LR either stays constant or decays (depends on lr_scheduler_type,
    # which defaults to "linear" decay to 0 over the remaining steps).

    weight_decay: float = 0.01
    # L2 regularisation coefficient.
    #
    # Before each weight update, the optimiser adds a penalty:
    #   effective_gradient += weight_decay × current_weight_value
    # This pulls weights toward zero, discouraging them from growing large.
    #
    # Why does this help? Large weights → model is very confident → tends to overfit.
    # Small weights → smoother, more general decisions.
    # 0.01 is the near-universal default in transformer fine-tuning.

    evaluation_strategy: str = "epoch"
    # WHEN to run the evaluation loop on the validation set.
    #
    # "epoch"  → evaluate once at the end of every training epoch.
    #            With 3 epochs, we get 3 evaluation snapshots.
    # "steps"  → evaluate every eval_steps training steps (more granular,
    #            better for catching early stopping opportunities, but slower).
    # "no"     → never evaluate (not recommended — you'd be training blind).

    save_strategy: str = "epoch"
    # WHEN to save a model checkpoint to disk.
    # Matched to evaluation_strategy="epoch" so every checkpoint has a
    # corresponding set of validation metrics.

    load_best_model_at_end: bool = True
    # After training finishes, automatically reload the checkpoint with the
    # BEST validation metric (F1 in our case), not necessarily the last one.
    #
    # Why? Epoch 3 might overfit slightly worse than Epoch 2. Without this,
    # you'd silently use the overfit weights. With this flag, Trainer picks
    # the best checkpoint automatically.

    metric_for_best_model: str = "f1"
    # The metric used to decide which checkpoint is "best".
    #
    # WHY F1 over accuracy?
    #   Accuracy = correct predictions / total predictions.
    #   If 50% of the dataset is spam, a model that ALWAYS predicts "not spam"
    #   gets 50% accuracy — but is completely useless!
    #
    #   F1 = 2 × (Precision × Recall) / (Precision + Recall)
    #   Precision = of emails flagged as spam, how many ACTUALLY are spam?
    #   Recall    = of all actual spam, how many did we CATCH?
    #
    #   F1 punishes both over-flagging (flagging ham as spam) and missing spam.
    #   It gives a single number that balances both concerns.

    logging_steps: int = 50
    # Print the training loss to the console every 50 optimiser steps.
    # Lower = more frequent logging (noisier but more insight during training).
    # Higher = less frequent (cleaner output, less visibility).

    report_to: str = "none"
    # Disable logging to external experiment trackers like Weights & Biases
    # or TensorBoard. For a local experiment, console output is enough.
    # Set to "wandb" if you want W&B integration (requires `pip install wandb`).

    fp16: bool = False
    # Whether to use 16-bit floating-point (half precision) arithmetic.
    #
    # fp16 halves memory usage and speeds up training significantly on NVIDIA
    # GPUs that have Tensor Cores (any RTX or A-series GPU).
    #
    # We set False because:
    #   1. Apple MPS (Metal) does not support fp16 training the same way CUDA does.
    #   2. GPT-2 (124M params) is small — fp32 training fits easily on a MacBook.
    #
    # On Google Colab (T4 GPU) or any NVIDIA GPU: set fp16=True for a free 2× speedup.


# ─────────────────────────────────────────────────────────────
#  3.  Data configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """
    Controls everything about how data is loaded and prepared.
    Keeping data settings separate means you can swap datasets without
    touching the model or training code.
    """

    dataset_path: str = "dataset/spam_ham_dataset.csv"
    # Path to the CSV file, relative to the project root directory
    # (fine-tuning-gpt2-classification/).
    # The project root is where you run scripts from.

    text_column: str = "text"
    # Name of the column in the CSV that contains the raw email text.
    # Our CSV has columns: ["label", "text", "label_num"]

    label_column: str = "label_num"
    # Name of the column with numeric labels.
    # label_num: 0 = ham (not spam), 1 = spam.
    # We use label_num (int) not label (string "spam"/"ham") because
    # PyTorch loss functions expect integer class indices, not strings.

    max_length: int = 128
    # Maximum number of tokens per email.
    # Anything longer is TRUNCATED (tail cut off).
    # Anything shorter is PADDED (filled with pad tokens on the right).
    #
    # ┌─ WHY 128 and not GPT-2's max of 1024? ────────────────────────────────┐
    # │                                                                         │
    # │  Attention computation is O(n²) in sequence length n.                  │
    # │  At n=1024: attention matrix = 1024 × 1024 = 1M multiplications.       │
    # │  At n=128:  attention matrix = 128 × 128  = 16K multiplications.       │
    # │  → 128 is 64× cheaper in attention computation.                        │
    # │                                                                         │
    # │  And spam classification? The key signals (suspicious words, urgency,   │
    # │  excessive caps, scam links) almost always appear in the first ~100      │
    # │  tokens. Keeping all 1024 adds noise, not signal.                      │
    # └─────────────────────────────────────────────────────────────────────────┘

    train_ratio: float = 0.8
    # 80% of the balanced dataset → training set.
    # 20% of the balanced dataset → validation set.
    # Standard split; can adjust to 0.9 if you have very little data.

    random_seed: int = 42
    # Fixed seed for all random operations (data shuffling, train/val split).
    # Same seed → same split every run → reproducible experiments.
    # You can compare two runs fairly only if their validation sets are identical.
    # 42 is a community convention (from The Hitchhiker's Guide to the Galaxy).

    max_samples_per_class: int = 1500
    # Cap on examples per class (spam / ham) to keep training fast.
    # Total: 1500 spam + 1500 ham = 3000 examples.
    # → 2400 train  + 600 val.
    #
    # The full dataset has ~50K spam and ~50K ham. Using all of it would
    # produce a better model but take ~15× longer to train.
    # For a portfolio experiment, 3000 examples is plenty to demonstrate
    # the technique and compare LoRA against the baseline.
    # Set to None (and update load_data) if you want the full dataset.
