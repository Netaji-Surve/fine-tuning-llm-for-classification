"""
train.py — The entry-point that wires all pieces together

This script is the "director": it calls functions from config, dataset,
model, and trainer in the right order to run a complete training experiment.

HOW TO RUN:
  From the project root (fine-tuning-gpt2-classification/):

  Option A — as a module (recommended):
    python -m src.qlora.train

  Option B — import and call from a notebook:
    from src.qlora.train import train
    train()

  Option C — override defaults for an experiment:
    from src.qlora.train import train
    from src.qlora.config import LoRAConfig, TrainingConfig
    train(
        lora_config=LoRAConfig(r=8),                      # try rank-8
        training_config=TrainingConfig(learning_rate=1e-4) # try lower LR
    )

WHY is this a separate file from trainer.py?
  trainer.py knows HOW to train (the mechanics of a Trainer object).
  train.py knows WHAT to train (the sequence of steps: load → build → run → save).
  Separating them makes each file easier to read and test independently.
"""

# os: standard library for operating-system utilities.
# We use os.path.join() to build file paths in a cross-platform way.
import os

# time: standard library for timing. We use it to measure total training time.
import time

# torch: needed to detect which device is available (CUDA / MPS / CPU)
import torch

# Import our three configuration dataclasses.
# The `or` pattern in train() lets callers omit any of these — we fill in defaults.
from .config  import LoRAConfig, TrainingConfig, DataConfig

# Import the data loading pipeline from dataset.py
from .dataset import load_data

# Import model + tokenizer factories and the diagnostic utility from model.py
from .model   import create_tokenizer, create_model, print_trainable_parameters

# Import the training loop builders from trainer.py
from .trainer import build_training_args, build_trainer


# ─────────────────────────────────────────────────────────────
#  Device detection
# ─────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Detects and returns the best available compute device as a string.

    Device priority:
      1. "cuda"  — NVIDIA GPU via CUDA. Fastest. Supports bitsandbytes 4-bit quant.
      2. "mps"   — Apple Silicon GPU via Metal. Fast. No bitsandbytes support.
      3. "cpu"   — Any machine. Always works. Training is slow (~10-30× slower).

    HuggingFace Trainer reads the available device automatically and moves
    the model + data there. We just print an informative message.
    """
    if torch.cuda.is_available():
        # torch.cuda.is_available() checks if CUDA drivers and a GPU are present.
        # torch.cuda.get_device_name(0) returns the name of the first GPU (index 0).
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        print(f"  Hardware: NVIDIA GPU — {name}")
        print(f"  Note: on CUDA you can enable 4-bit quantization (full QLoRA)")
        print(f"        by setting load_in_4bit=True in model.py's LoraConfig.")

    elif torch.backends.mps.is_available():
        # torch.backends.mps.is_available() checks for Apple's Metal Performance Shaders.
        # MPS is available on M1, M2, M3 (and later) Macs running macOS 12.3+.
        device = "mps"
        print(f"  Hardware: Apple Silicon (Metal / MPS)")
        print(f"  Running LoRA without 4-bit quantization.")
        print(f"  Reason: bitsandbytes (which does NF4 quant) requires CUDA.")
        print(f"  → For full QLoRA (with quantization), use Google Colab (free T4 GPU)")
        print(f"    or any machine with an NVIDIA GPU.")

    else:
        device = "cpu"
        print(f"  Hardware: CPU only")
        print(f"  Warning: training on CPU is very slow. Expect 30-60 minutes for 3 epochs.")
        print(f"  Tip: use Google Colab for free GPU access.")

    return device


# ─────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────

def train(
    lora_config:     LoRAConfig     = None,
    training_config: TrainingConfig = None,
    data_config:     DataConfig     = None,
) -> dict:
    """
    Runs the full training pipeline: configure → load → build → train → save.

    All three arguments are optional. Pass None (or omit them) to use the
    sensible defaults defined in config.py. Pass a modified config to run
    a different experiment without touching any other file.

    Args:
        lora_config:     Controls LoRA adapter (rank, alpha, target layers)
        training_config: Controls training loop (LR, epochs, batch size, etc.)
        data_config:     Controls data loading (path, max_length, split ratio)

    Returns:
        A dict with training metrics: {"train_loss": ..., "val_f1": ..., ...}
        Useful when calling train() from a notebook to log multiple experiments.
    """

    # ── Fill in defaults for any configs not provided ─────────────────────────
    # The `or` operator: if lora_config is None (falsy), create a new default one.
    # This lets callers do: train()                         ← all defaults
    #                       train(lora_config=LoRAConfig(r=8))  ← override rank only
    lora_config     = lora_config     or LoRAConfig()
    training_config = training_config or TrainingConfig()
    data_config     = data_config     or DataConfig()

    print("=" * 60)
    print("  QLoRA Spam Classifier — Training Run")
    print("=" * 60)

    # ── Step 1: Detect hardware ────────────────────────────────────────────────
    print("\n[1/5] Detecting hardware...")
    device = get_device()

    # ── Step 2: Load tokenizer ─────────────────────────────────────────────────
    print("\n[2/5] Loading tokenizer...")
    # The tokenizer must be the SAME one used to pre-train the model.
    # Using a different tokenizer would give completely different token IDs,
    # and the model's embedding table would map them to meaningless vectors.
    tokenizer = create_tokenizer("gpt2")

    # ── Step 3: Load and prepare data ─────────────────────────────────────────
    print("\n[3/5] Loading and preparing data...")
    train_dataset, val_dataset = load_data(data_config, tokenizer)

    # ── Step 4: Create model with LoRA adapters ────────────────────────────────
    print("\n[4/5] Creating model with LoRA adapters...")
    model = create_model(lora_config, model_name="gpt2")

    # Print the parameter breakdown — this is the key diagnostic that shows
    # how LoRA achieves parameter efficiency.
    print_trainable_parameters(model)

    # ── Step 5: Build Trainer and run training ─────────────────────────────────
    print("\n[5/5] Starting training...")

    # Build the TrainingArguments object (wraps our config into HuggingFace format)
    training_args = build_training_args(training_config)

    # Assemble the Trainer (links model, data, args, and metrics function)
    trainer = build_trainer(model, train_dataset, val_dataset, training_args)

    # Start the clock so we can report total training time.
    t_start = time.time()

    # trainer.train() runs the ENTIRE training loop:
    #   for epoch in range(num_epochs):
    #       for batch in train_dataloader: forward → backward → update → log
    #       evaluate on val_dataset → call compute_metrics → log metrics
    #       save checkpoint
    #   if load_best_model_at_end: reload best checkpoint
    #
    # Returns a TrainOutput with:
    #   .global_step       — total number of optimiser steps
    #   .training_loss     — average loss over the whole training run
    #   .metrics           — dict with timing and loss info
    result = trainer.train()

    t_end = time.time()
    elapsed = t_end - t_start

    # ── Save the final model ───────────────────────────────────────────────────
    # model.save_pretrained() saves only the LoRA adapter weights (A and B matrices),
    # NOT the full GPT-2 backbone (which is frozen and unchanged from the original).
    #
    # What gets saved:
    #   adapter_config.json   — records r, alpha, target_modules etc.
    #   adapter_model.safetensors — the actual A and B weight tensors (~few MB)
    #
    # What does NOT get saved:
    #   The 124M frozen GPT-2 weights — you reload those from HuggingFace on demand.
    #
    # This is a huge advantage over full fine-tuning:
    #   Full fine-tuning saves: ~500 MB (entire model)
    #   LoRA saves:             ~5–10 MB (adapters only)
    save_dir = os.path.join(training_config.output_dir, "final_model")
    # os.makedirs creates the directory and any missing parent directories.
    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)     # Save LoRA adapter weights + config
    tokenizer.save_pretrained(save_dir) # Save tokenizer vocab (needed for inference)

    # ── Final report ──────────────────────────────────────────────────────────
    # Run one final evaluation on the validation set to get the best metrics.
    final_metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time          : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Training loss       : {result.training_loss:.4f}")
    print(f"  Val accuracy        : {final_metrics.get('eval_accuracy', 'N/A')}")
    print(f"  Val F1              : {final_metrics.get('eval_f1', 'N/A')}")
    print(f"  Val precision       : {final_metrics.get('eval_precision', 'N/A')}")
    print(f"  Val recall          : {final_metrics.get('eval_recall', 'N/A')}")
    print(f"  Model saved to      : {save_dir}/")
    print(f"\n  Adapter file size is ~few MB (vs ~500 MB for full fine-tuning).")
    print(f"  To load for inference:")
    print(f"    from peft import PeftModel")
    print(f"    from transformers import GPT2ForSequenceClassification")
    print(f"    base = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)")
    print(f"    model = PeftModel.from_pretrained(base, '{save_dir}')")
    print("=" * 60)

    # Return a summary dict so notebook cells can store and compare results.
    return {
        "training_loss": result.training_loss,
        "val_accuracy":  final_metrics.get("eval_accuracy"),
        "val_f1":        final_metrics.get("eval_f1"),
        "val_precision": final_metrics.get("eval_precision"),
        "val_recall":    final_metrics.get("eval_recall"),
        "elapsed_sec":   elapsed,
        "lora_r":        lora_config.r,
        "lora_alpha":    lora_config.lora_alpha,
        "learning_rate": training_config.learning_rate,
    }


# ─────────────────────────────────────────────────────────────
#  Script entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This block only runs when the script is executed DIRECTLY:
    #   python -m src.qlora.train
    #
    # It does NOT run when this module is imported in a notebook:
    #   from src.qlora.train import train   ← __name__ == "src.qlora.train", not "__main__"
    #
    # This is the standard Python idiom to make a module both importable
    # (as a library) and runnable (as a script).
    results = train()
    print("\nReturned metrics dict:", results)
