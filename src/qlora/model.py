"""
model.py — GPT-2 base model + LoRA adapters

This file does two things:
  1. Loads GPT-2 with a classification head  (create_model)
  2. Wraps it with LoRA adapters              (via peft's get_peft_model)

─────────────────────────────────────────────────────────────────────────────
DEEP DIVE: How LoRA adapters work inside a layer
─────────────────────────────────────────────────────────────────────────────

Consider GPT-2's c_attn layer, a Conv1D (functionally a linear layer):
  Input  x : shape (batch, seq_len, 768)   — the hidden states
  Weight W₀: shape (768, 2304)             — the frozen pre-trained weight
  Output y : shape (batch, seq_len, 2304)  — Q/K/V combined

Without LoRA:
  y = x W₀ᵀ                      (just the original computation)

With LoRA:
  y = x W₀ᵀ  +  (α/r) · x Aᵀ Bᵀ
      ───────    ─────────────────
      frozen        LoRA adapter

Where:
  A : shape (r, 768)    — the "down-projection"  (r << 768)
  B : shape (2304, r)   — the "up-projection"
  α/r is the scaling factor (lora_alpha / r)

Parameter count:
  Original W₀:  768 × 2304  = 1,769,472  ← FROZEN, not trained
  LoRA A:        16 × 768   =    12,288  ┐
  LoRA B:      2304 × 16    =    36,864  ┘ = 49,152 trained  (97% fewer!)

INITIALISATION:
  A is sampled from a random normal distribution (gives gradients from step 1).
  B is initialised to ALL ZEROS (so the adapter output is 0 at the start,
  meaning the model behaves identically to the frozen base model at the
  beginning of training — a stable starting point).

─────────────────────────────────────────────────────────────────────────────
DEEP DIVE: What is QLoRA (vs plain LoRA)?
─────────────────────────────────────────────────────────────────────────────

LoRA  = freeze base model + train low-rank adapters.
QLoRA = freeze base model + QUANTIZE IT TO 4 BITS + train low-rank adapters.

The "Q" (Quantization) step from Dettmers et al. (2023):
  1. NF4 (NormalFloat 4-bit): store each weight as one of 16 possible values
     chosen to match the normal distribution of pre-trained weights.
     768 × 2304 floats × 32 bits → × 4 bits = 8× memory reduction per layer.

  2. Double Quantization: even the 32-bit "quantization constants" are
     quantized again (to 8-bit), saving another ~0.5 bits/param on average.

  3. Paged Optimizers: use CPU RAM as overflow memory for the optimizer states
     (which can be 8× the model size) to handle memory spikes.

RESULT: a 7B-parameter model that needed 28 GB GPU memory with full fp32
can be fine-tuned on a 24 GB GPU using QLoRA.

ON APPLE MPS (our setup):
  `bitsandbytes` (the library that does NF4 quantization) requires CUDA.
  It does NOT work on Apple Metal/MPS. So we run plain LoRA (no quantization).
  The technique is identical — only the memory saving from 4-bit is missing.
  On a CUDA machine or Google Colab, set load_in_4bit=True to get full QLoRA.
─────────────────────────────────────────────────────────────────────────────
"""

# torch: needed to move the model to the correct device (MPS / CUDA / CPU)
import torch

# GPT2ForSequenceClassification: GPT-2 backbone + a linear "score" head on top.
# GPT2Tokenizer: the tokenizer that converts text to GPT-2 token IDs.
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

# peft: "Parameter-Efficient Fine-Tuning" — a HuggingFace library that
# implements LoRA, Prefix Tuning, Adapter Layers, and other PEFT methods.
# LoraConfig: a dataclass that tells peft which layers to adapt and how.
# get_peft_model: a function that wraps a base model with LoRA adapters.
# TaskType: an enum specifying the model's training objective.
from peft import LoraConfig, get_peft_model, TaskType

# Our own configuration dataclass
from .config import LoRAConfig


# ─────────────────────────────────────────────────────────────
#  Tokenizer factory
# ─────────────────────────────────────────────────────────────

def create_tokenizer(model_name: str = "gpt2") -> GPT2Tokenizer:
    """
    Loads the GPT-2 tokenizer and patches it for classification use.

    WHY does GPT-2 need patching?
      GPT-2 was designed for text GENERATION: given a prefix, predict the
      next token, then the next, then the next.
      Generation produces variable-length output — no padding needed.

      For CLASSIFICATION, we process a whole batch of sequences in parallel.
      All sequences in a batch must be the SAME length (so PyTorch can stack
      them into a single tensor). We achieve same-length by padding shorter
      sequences with a special [PAD] token.

      GPT-2's tokenizer has no [PAD] token defined. The simplest fix:
      reuse the [EOS] (end-of-sequence) token as [PAD].
      EOS has token ID 50256 in GPT-2's vocabulary.

      The attention_mask (1 for real, 0 for padding) ensures the model
      ignores the padding positions in attention — so using EOS as PAD
      does not corrupt the model's understanding of "end of sequence".

    Args:
        model_name: HuggingFace model ID. "gpt2" = base GPT-2 (124M params).
                    Other options: "gpt2-medium" (345M), "gpt2-large" (774M).

    Returns:
        A GPT2Tokenizer with pad_token set to eos_token.
    """

    # .from_pretrained() downloads (or loads from cache) the tokenizer vocab
    # and merges files for the given model. For GPT-2 this is ~1 MB.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token to be the same as the end-of-sequence token.
    # After this line, tokenizer.pad_token == "<|endoftext|>" and
    # tokenizer.pad_token_id == 50256.
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Tokenizer loaded: vocab size = {tokenizer.vocab_size:,} tokens")
    print(f"  Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    return tokenizer


# ─────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────

def create_model(
    lora_config: LoRAConfig,
    model_name: str = "gpt2"
) -> torch.nn.Module:
    """
    Loads GPT-2 with a classification head and wraps it with LoRA adapters.

    The returned model has:
      - All original GPT-2 weights FROZEN (requires_grad = False)
      - LoRA A and B matrices added to each target layer (requires_grad = True)
      - The classification "score" head (768 → 2) fully trainable

    Args:
        lora_config: Our LoRAConfig dataclass (r, alpha, dropout, target_modules)
        model_name:  HuggingFace model ID string

    Returns:
        A PeftModel (wrapping GPT2ForSequenceClassification) with LoRA adapters
    """

    # ── STEP 1: Load the base model ───────────────────────────────────────────
    # GPT2ForSequenceClassification is GPT-2 with ONE extra layer:
    # a linear "score" layer that maps hidden_state → class logits.
    #
    # Internally, it looks like:
    #   GPT-2 backbone (12 transformer blocks)  →  hidden state (batch, seq, 768)
    #                                                      ↓
    #   Take hidden state at the LAST non-padding token    ↓
    #                                                      ↓
    #   score = nn.Linear(768, num_labels, bias=False)     ↓
    #                                                      ↓
    #   output logits: (batch, 2)   [logit_ham, logit_spam]
    #
    # num_labels=2: binary classification (ham or spam).
    print(f"\nLoading base model: {model_name}")
    base_model = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Tell the model what the padding token ID is.
    # GPT-2 uses the LAST non-padding token's hidden state for classification.
    # Without knowing pad_token_id, it can't determine which token is "last real".
    base_model.config.pad_token_id = base_model.config.eos_token_id

    print(f"  Base model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters")

    # ── STEP 2: Build the peft LoraConfig ─────────────────────────────────────
    # peft's LoraConfig is SEPARATE from our LoRAConfig dataclass.
    # We translate from our config into peft's expected format here.
    peft_config = LoraConfig(

        # task_type tells peft which parts of the model to treat specially.
        # SEQ_CLS = sequence classification. peft will:
        #   - Not apply LoRA to the final classification head (score layer)
        #   - Instead, the score layer goes into modules_to_save (fully trainable)
        task_type=TaskType.SEQ_CLS,

        # r: the rank. Passed directly to peft.
        r=lora_config.r,

        # lora_alpha: the scaling factor. The effective scale = lora_alpha / r.
        lora_alpha=lora_config.lora_alpha,

        # lora_dropout: dropout probability on the adapter output.
        lora_dropout=lora_config.lora_dropout,

        # target_modules: list of substrings. peft scans ALL named modules in the
        # model. Any module whose name CONTAINS one of these strings gets a
        # LoRA adapter. GPT-2's attention modules are named:
        #   transformer.h.0.attn.c_attn   ← "c_attn" is a substring → match!
        #   transformer.h.0.attn.c_proj   ← "c_proj" is a substring → match!
        #   transformer.h.0.mlp.c_fc      ← "c_fc" is a substring → only if added
        #   ... (repeated for all 12 blocks)
        target_modules=lora_config.target_modules,

        # bias: whether to train bias terms alongside A and B.
        bias=lora_config.bias,

        # modules_to_save: these modules are NOT wrapped with LoRA.
        # Instead, the ENTIRE module is made trainable and its weights are
        # saved separately when we call model.save_pretrained().
        # "score" = the classification head (nn.Linear, 768 → 2).
        # It's task-specific (only makes sense for spam classification),
        # so we train it fully. It's tiny: 768 × 2 = 1,536 parameters.
        modules_to_save=["score"],
    )

    # ── STEP 3: Wrap the base model with LoRA ─────────────────────────────────
    # get_peft_model() performs the following operations on each matched layer:
    #
    #   For a target Conv1D layer (e.g., c_attn with weight shape 768 × 2304):
    #     1. Freeze the original weight: param.requires_grad = False
    #     2. Create lora_A: nn.Linear(768, r, bias=False) with random init
    #     3. Create lora_B: nn.Linear(r, 2304, bias=False) with ZERO init
    #     4. Replace the original layer with a LoraLayer that computes:
    #           output = original(x)  +  scaling * lora_B(lora_A(x))
    #        where scaling = lora_alpha / r
    #
    # The result is a PeftModel — a thin wrapper around the base model that
    # knows how to save/load only the adapter weights.
    print(f"\nAttaching LoRA adapters...")
    print(f"  Target modules: {lora_config.target_modules}")
    print(f"  Rank (r): {lora_config.r}  |  Alpha: {lora_config.lora_alpha}  |  Scale: {lora_config.lora_alpha / lora_config.r:.1f}")
    model = get_peft_model(base_model, peft_config)

    return model


# ─────────────────────────────────────────────────────────────
#  Parameter count utility
# ─────────────────────────────────────────────────────────────

def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Prints a summary of trainable vs. frozen parameters in the model.

    This is the single most important diagnostic for LoRA:
    it proves how few parameters we're actually training.

    EXPECTED OUTPUT (GPT-2 base, r=16, target=[c_attn, c_proj]):
      Trainable params:  592,900  (~0.47%)
      All params:       124,443,648
      → 210x fewer parameters than full fine-tuning!
    """

    trainable = 0  # Running total of parameters with requires_grad=True
    total     = 0  # Running total of ALL parameters

    # model.parameters() is a generator that yields every nn.Parameter tensor.
    # This includes all layers: embeddings, attention, FFN, LoRA adapters, etc.
    for param in model.parameters():

        # param.numel() = "number of elements" = total scalar values in the tensor.
        # For a 2D weight matrix of shape (768, 2304): numel() = 768 × 2304 = 1,769,472
        count = param.numel()
        total += count

        # requires_grad=True means gradients will be computed for this parameter
        # during backpropagation, and the optimizer will update its values.
        # LoRA A and B matrices: requires_grad=True  (we train these)
        # Original W₀ weights:   requires_grad=False (frozen, never updated)
        if param.requires_grad:
            trainable += count

    # Compute what percentage of total parameters are being trained.
    pct = 100.0 * trainable / total

    print("\n" + "=" * 55)
    print(f"  MODEL PARAMETER SUMMARY")
    print("=" * 55)
    print(f"  Trainable parameters : {trainable:>12,}  ({pct:.4f}%)")
    print(f"  Frozen parameters    : {total - trainable:>12,}  ({100-pct:.4f}%)")
    print(f"  Total parameters     : {total:>12,}  (100%)")
    print("=" * 55)
    print(f"\n  Full fine-tuning would train all {total:,} params.")
    print(f"  LoRA trains only {trainable:,} params.")
    print(f"  That is a {total // trainable}x reduction in trainable parameters.")
    print(f"\n  Memory saving (approximate):")
    # Adam optimizer stores: gradient + first moment + second moment = 3 × param size
    # For fp32, each param = 4 bytes
    full_opt_mb  = total     * 4 * 3 / 1e6
    lora_opt_mb  = trainable * 4 * 3 / 1e6
    print(f"    Full fine-tuning optimizer states: ~{full_opt_mb:.0f} MB")
    print(f"    LoRA optimizer states:             ~{lora_opt_mb:.0f} MB")
    print(f"    Saving: ~{full_opt_mb - lora_opt_mb:.0f} MB\n")
