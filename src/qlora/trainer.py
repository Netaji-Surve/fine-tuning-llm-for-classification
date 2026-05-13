"""
trainer.py — Building and running the HuggingFace training loop

WHY use HuggingFace Trainer instead of writing a manual PyTorch loop?
──────────────────────────────────────────────────────────────────────
A manual training loop looks like this:

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()          # clear old gradients
            outputs = model(**batch)       # forward pass
            loss = outputs.loss            # compute loss
            loss.backward()               # backward pass (compute gradients)
            optimizer.step()              # update weights
            scheduler.step()             # update learning rate
            log_metrics(loss)            # print progress
        evaluate(model, val_loader)      # validation
        save_checkpoint(model)          # save

That's ~30 lines of boilerplate, and it doesn't include:
  - Moving data to the right device (MPS/CUDA/CPU)
  - Mixed precision (fp16) handling
  - Gradient clipping (prevent exploding gradients)
  - Best checkpoint tracking
  - Resume from checkpoint
  - Distributed training across multiple GPUs

HuggingFace Trainer handles ALL of this. We just provide:
  model + training data + validation data + settings + metrics function.
"""

# numpy: numerical Python library. We use it to manipulate arrays of predictions
# before computing metrics (numpy is faster than plain Python lists for this).
import numpy as np

# Trainer: the HuggingFace class that encapsulates the full training loop.
# TrainingArguments: a dataclass that configures every aspect of Trainer.
from transformers import Trainer, TrainingArguments

# Metric functions from scikit-learn:
# accuracy_score: fraction of predictions that exactly match labels
# f1_score:       harmonic mean of precision and recall (best single metric for classification)
# precision_score: of items predicted positive, what fraction actually are positive
# recall_score:    of actual positives, what fraction did we predict as positive
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Type hint: Tuple[A, B] means "this function returns a tuple of types A and B"
from typing import Tuple

# Import our config dataclass
from .config import TrainingConfig


# ─────────────────────────────────────────────────────────────
#  Metrics function
# ─────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    """
    Converts raw model outputs into human-readable classification metrics.

    HuggingFace Trainer calls this function at the end of EVERY evaluation run.
    It receives an EvalPrediction object (named eval_pred here) that contains
    two numpy arrays:

      eval_pred.predictions  — raw logits, shape (num_val_examples, num_labels)
                               e.g. [[2.1, -0.3],   ← "strongly ham"
                                     [0.1,  4.2],   ← "strongly spam"
                                     [-0.5, 0.3]]   ← "weakly spam"

      eval_pred.label_ids    — ground-truth labels, shape (num_val_examples,)
                               e.g. [0, 1, 1]

    We must return a dict mapping metric name → float.
    The key matching `metric_for_best_model` in TrainingConfig drives
    best-checkpoint selection.

    ─────────────────────────────────────────────────────────────────────────
    CONCEPT: What are logits?
    ─────────────────────────────────────────────────────────────────────────
    The model outputs RAW scores (logits), not probabilities.
    Logit = the value BEFORE applying softmax.

    To convert logits → probabilities:
      softmax([2.1, -0.3]) = [e^2.1 / (e^2.1 + e^-0.3), e^-0.3 / (e^2.1 + e^-0.3)]
                           ≈ [0.92, 0.08]   ← 92% ham, 8% spam

    For classification, we don't actually need probabilities — we just need
    to know WHICH class has the higher score. argmax gives us that directly:
      argmax([2.1, -0.3]) = 0  → predicted ham
      argmax([0.1,  4.2]) = 1  → predicted spam
    ─────────────────────────────────────────────────────────────────────────
    """

    # Unpack the EvalPrediction tuple into two variables.
    # logits: shape (num_val_examples, 2)
    # labels: shape (num_val_examples,) — integer 0 or 1
    logits, labels = eval_pred

    # np.argmax(array, axis=1):
    #   axis=1 means "find the max along dimension 1 (the columns)".
    #   For each row (each example), returns the index of the maximum value.
    #   [[2.1, -0.3],   → index 0 (ham wins)
    #    [0.1,  4.2]]   → index 1 (spam wins)
    #   Result: [0, 1]  — the predicted class for each example
    predictions = np.argmax(logits, axis=1)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    # accuracy_score(true_labels, predicted_labels)
    # = number of correct predictions / total predictions
    # = (TP + TN) / (TP + TN + FP + FN)
    #
    # Limitation: accuracy is misleading on imbalanced data.
    # On a dataset that is 90% ham, predicting "ham" every time gives 90% accuracy
    # but 0% spam recall. This is why we also report F1.
    accuracy = accuracy_score(labels, predictions)

    # ── F1 Score ───────────────────────────────────────────────────────────────
    # F1 is the harmonic mean of Precision and Recall.
    #
    # First, understand the 2×2 confusion matrix for spam detection:
    #
    #                      Predicted: Ham    Predicted: Spam
    #   Actual: Ham   →       TN               FP (false alarm)
    #   Actual: Spam  →       FN (missed!)     TP
    #
    # Precision = TP / (TP + FP)
    #   "Of all emails we flagged as spam, what fraction actually IS spam?"
    #   Low precision = too many false alarms (ham incorrectly flagged)
    #
    # Recall = TP / (TP + FN)
    #   "Of all actual spam emails, what fraction did we CATCH?"
    #   Low recall = too many missed spam (spam slips through)
    #
    # F1 = 2 × (Precision × Recall) / (Precision + Recall)
    #   If either precision or recall is very low, F1 is pulled down.
    #   Only a model that is BOTH precise AND has high recall gets a high F1.
    #
    # average="binary": compute F1 for the positive class (spam=1) only.
    # This is appropriate for binary classification.
    # Alternative: average="macro" (average of F1 for each class separately).
    f1 = f1_score(labels, predictions, average="binary")

    # ── Precision ─────────────────────────────────────────────────────────────
    # zero_division=0: if the model never predicts spam (TP + FP = 0),
    # return 0 instead of raising a ZeroDivisionError.
    precision = precision_score(labels, predictions, average="binary", zero_division=0)

    # ── Recall ────────────────────────────────────────────────────────────────
    recall = recall_score(labels, predictions, average="binary")

    # Return a dict. Keys become column names in the Trainer's evaluation log.
    # The key "f1" must match `metric_for_best_model="f1"` in TrainingConfig.
    return {
        "accuracy":  round(accuracy,  4),
        "f1":        round(f1,        4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
    }


# ─────────────────────────────────────────────────────────────
#  TrainingArguments factory
# ─────────────────────────────────────────────────────────────

def build_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Converts our TrainingConfig dataclass into a HuggingFace TrainingArguments object.

    TrainingArguments controls EVERYTHING the Trainer does:
      - How long to train (epochs)
      - How fast to learn (learning rate, warmup, weight decay)
      - When to evaluate and save (every epoch)
      - What to log (steps, destination)
      - Hardware settings (fp16, device placement)
    """

    print("\nBuilding TrainingArguments:")
    print(f"  Output directory   : {config.output_dir}")
    print(f"  Epochs             : {config.num_train_epochs}")
    print(f"  Train batch size   : {config.per_device_train_batch_size}")
    print(f"  Learning rate      : {config.learning_rate}")
    print(f"  Warmup steps       : {config.warmup_steps}")
    print(f"  Weight decay       : {config.weight_decay}")
    print(f"  Best model metric  : {config.metric_for_best_model}")

    return TrainingArguments(
        # Where to write checkpoints and the final model
        output_dir=config.output_dir,

        # Total number of training epochs
        num_train_epochs=config.num_train_epochs,

        # Batch size for training (per device)
        per_device_train_batch_size=config.per_device_train_batch_size,

        # Batch size for evaluation — can be larger since no backprop
        per_device_eval_batch_size=config.per_device_eval_batch_size,

        # Starting learning rate for the AdamW optimizer.
        # AdamW is Adam + weight decay decoupled from the gradient update.
        learning_rate=config.learning_rate,

        # Steps of linear LR warmup (0 → learning_rate over warmup_steps steps)
        warmup_steps=config.warmup_steps,

        # L2 regularisation strength (applied to all non-bias, non-LayerNorm weights)
        weight_decay=config.weight_decay,

        # Run evaluation at the end of each epoch
        eval_strategy=config.evaluation_strategy,

        # Save a checkpoint at the end of each epoch
        save_strategy=config.save_strategy,

        # After training, reload the checkpoint with the best validation metric
        load_best_model_at_end=config.load_best_model_at_end,

        # The metric to optimise when choosing the "best" checkpoint
        metric_for_best_model=config.metric_for_best_model,

        # True because higher F1 = better (vs loss where lower = better)
        greater_is_better=True,

        # Log training loss to console every N steps
        logging_steps=config.logging_steps,

        # Disable W&B / TensorBoard / etc. — console only
        report_to=config.report_to,

        # fp16 training: disabled for MPS compatibility (see config.py for details)
        fp16=config.fp16,

        # Keep only the best N checkpoints to save disk space.
        # With 3 epochs, this keeps the 2 best checkpoints.
        save_total_limit=2,
    )


# ─────────────────────────────────────────────────────────────
#  Trainer factory
# ─────────────────────────────────────────────────────────────

def build_trainer(
    model,
    train_dataset,
    val_dataset,
    training_args: TrainingArguments
) -> Trainer:
    """
    Assembles all pieces into a HuggingFace Trainer, ready for trainer.train().

    The Trainer ties together:
      model        — what to train
      training_args — how to train (LR, epochs, device, etc.)
      train_dataset — the data to train on
      eval_dataset  — the data to validate on after each epoch
      compute_metrics — how to score the model on the validation set

    ─────────────────────────────────────────────────────────────────────────
    WHAT HAPPENS INSIDE trainer.train():
    ─────────────────────────────────────────────────────────────────────────

    Trainer creates two DataLoaders (one for train, one for eval).
    DataLoader is a PyTorch class that:
      - Wraps our SpamDataset
      - Shuffles the training indices at the start of each epoch
      - Calls SpamDataset.__getitem__(idx) for each sample
      - Stacks samples into batches of shape (batch_size, max_length)
      - Moves tensors to the target device (MPS/CUDA/CPU)

    Then for each step:
      1. batch = next(train_dataloader)
         batch is a dict: {input_ids: (8,128), attention_mask: (8,128), labels: (8,)}

      2. outputs = model(**batch)
         model(**batch) unpacks the dict as keyword arguments to model.forward()
         outputs.loss = CrossEntropyLoss(outputs.logits, batch["labels"])
              CrossEntropyLoss = -log(softmax(logit_for_correct_class))
              It penalises wrong or uncertain predictions.

      3. outputs.loss.backward()
         PyTorch computes the gradient ∂loss/∂θ for every θ where requires_grad=True.
         For LoRA: gradients flow into lora_A and lora_B matrices only.
         The frozen W₀ receives no gradient (the backward pass stops there).

      4. optimizer.step()
         AdamW updates each trainable parameter:
           m = β₁·m + (1-β₁)·gradient           ← first moment (mean)
           v = β₂·v + (1-β₂)·gradient²           ← second moment (variance)
           θ ← θ - lr · m̂ / (√v̂ + ε) - lr·λ·θ  ← AdamW update with weight decay

      5. optimizer.zero_grad()
         Clear gradients so they don't accumulate across steps.
    ─────────────────────────────────────────────────────────────────────────
    """

    return Trainer(
        model=model,                     # The PeftModel (GPT-2 + LoRA adapters)
        args=training_args,              # All the training settings
        train_dataset=train_dataset,     # SpamDataset wrapping training examples
        eval_dataset=val_dataset,        # SpamDataset wrapping validation examples
        compute_metrics=compute_metrics, # Called after each eval loop
    )
