"""
dataset.py — Turning raw CSV data into PyTorch tensors

The gap between "a CSV file on disk" and "numbers the GPU can process"
is bigger than it looks. This file bridges that gap in three stages:

  Stage 1 — load_data():      CSV → balanced DataFrame → train/val lists
  Stage 2 — SpamDataset:      lists → a PyTorch Dataset object
  Stage 3 — __getitem__():    one raw string → dict of 3 tensors (on demand)

Stage 3 happens LAZILY — each email is tokenized only when the DataLoader
asks for it (i.e., during the training loop), not all at once upfront.
This is memory-efficient: you never hold all 2400 tokenized sequences in RAM.
"""

# pandas: the standard library for reading CSV files and manipulating tables.
# A DataFrame is like a spreadsheet: rows are samples, columns are fields.
import pandas as pd

# torch: the core PyTorch library. Everything ultimately becomes a torch.Tensor —
# a multi-dimensional array stored in memory (RAM or GPU VRAM) that supports
# automatic differentiation.
import torch

# Dataset: an abstract base class from PyTorch. Any class that:
#   1. inherits from Dataset
#   2. implements __len__() → int
#   3. implements __getitem__(idx) → any
# can be plugged into a DataLoader, which handles batching, shuffling, and
# parallel data loading automatically.
from torch.utils.data import Dataset

# GPT2Tokenizer: converts raw text strings into integer token IDs that GPT-2
# understands. GPT-2 uses Byte-Pair Encoding (BPE): it splits text into
# "subword" units learned from a large text corpus.
# e.g., "unhappy" might become ["un", "happy"] → [403, 4083]
from transformers import GPT2Tokenizer

# train_test_split: a scikit-learn function that splits a list (or array)
# into two subsets. We use it to create the train / validation split.
from sklearn.model_selection import train_test_split

# Tuple: a type hint — Tuple[A, B] means "a tuple containing type A and type B"
from typing import Tuple

# Import our configuration dataclass from the sibling module.
# The dot in .config means "look in the same package (qlora)" not globally.
from .config import DataConfig


# ─────────────────────────────────────────────────────────────
#  The Dataset class
# ─────────────────────────────────────────────────────────────

class SpamDataset(Dataset):
    """
    A PyTorch Dataset wrapping a list of emails and their labels.

    HOW IT FITS IN THE BIGGER PICTURE:
      SpamDataset → DataLoader → Trainer training loop → model

    DataLoader wraps SpamDataset and:
      - Calls __len__() to know total samples
      - Calls __getitem__(i) for each index
      - Stacks the returned dicts into batches (tensors of shape [batch, seq_len])
      - Optionally shuffles indices each epoch
      - Optionally uses multiple worker processes to load data in parallel
    """

    def __init__(
        self,
        texts: list,              # List of raw email strings, e.g. ["Free money!!!", ...]
        labels: list,             # List of integer labels, e.g. [1, 0, 1, 0, ...]
        tokenizer: GPT2Tokenizer, # The tokenizer instance — shared with the model
        max_length: int = 128     # Truncate/pad all sequences to this many tokens
    ):
        # Store the inputs as instance attributes.
        # `self.` binds the value to THIS specific Dataset object.
        # Other methods (like __getitem__) access them via `self.texts` etc.
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the total number of examples in this dataset.

        DataLoader calls this once at the start to know how many batches
        to create per epoch:
          num_batches = ceil( len(dataset) / batch_size )
          e.g., ceil(2400 / 8) = 300 batches per epoch
        """
        # len(self.texts) = number of elements in the list
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns ONE training example as a dictionary of PyTorch tensors.

        DataLoader calls this method with integer indices (0, 1, 2, ...).
        It then stacks multiple single-example dicts into a batch:
          batch["input_ids"] shape: (batch_size, max_length) = (8, 128)

        Args:
            idx: the integer index of the example to retrieve

        Returns:
            A dict with three tensors:
              "input_ids"      — the token IDs for the email
              "attention_mask" — 1 for real tokens, 0 for padding
              "labels"         — the ground-truth class (0=ham, 1=spam)
        """

        # Retrieve the raw text at this index and ensure it's a string.
        # str() handles edge cases where the CSV might have stored NaN (float).
        text = str(self.texts[idx])

        # ── TOKENIZATION ──────────────────────────────────────────────────────
        # self.tokenizer() calls the tokenizer like a function. It does:
        #
        #   1. SPLIT into subword tokens using BPE vocabulary
        #      "Congratulations! You won $1000 today!"
        #       → ["Congrat", "ulations", "!", " You", " won", " $", "1000", ...]
        #
        #   2. LOOK UP each token in the 50,257-word vocabulary
        #      → [34, 19082, 0, 921, 1839, 720, 13454, ...]  (integer IDs)
        #
        #   3. TRUNCATE if the sequence exceeds max_length=128 tokens
        #      (the tail is cut off — the beginning is kept)
        #
        #   4. PAD if the sequence is shorter than max_length
        #      GPT-2 has no native pad token, so we set pad_token = eos_token (ID 50256).
        #      Short emails get [50256, 50256, ...] appended to reach length 128.
        #
        #   5. Build the ATTENTION MASK:
        #      A binary tensor the same length as input_ids:
        #        1 = real token (the model should attend to this)
        #        0 = padding token (the model should IGNORE this)
        #      This prevents the model from "reading" the padding as meaningful content.
        #
        #   6. Return PyTorch tensors (return_tensors="pt")
        encoding = self.tokenizer(
            text,
            truncation=True,      # Cut off tokens beyond max_length
            max_length=self.max_length,
            padding="max_length", # Pad shorter sequences to exactly max_length
            return_tensors="pt"   # Return torch.Tensor (not list or numpy array)
        )

        # encoding is a BatchEncoding dict with:
        #   encoding["input_ids"]      shape: (1, 128)  ← extra batch dim of size 1
        #   encoding["attention_mask"] shape: (1, 128)
        #
        # We call .squeeze(0) to remove that extra dimension:
        #   (1, 128) → (128,)
        #
        # WHY does the tokenizer add a batch dim?
        #   It's designed for batched calls: tokenizer(["text1", "text2"]) → (2, 128).
        #   When given a single string, it still wraps it in a batch of size 1.
        #   We undo that here so DataLoader can re-batch correctly.
        input_ids      = encoding["input_ids"].squeeze(0)       # shape: (128,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # shape: (128,)

        # Convert the Python integer label to a scalar PyTorch tensor.
        # dtype=torch.long = 64-bit integer, required by CrossEntropyLoss.
        # (CrossEntropyLoss expects class indices as integers, not floats.)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # shape: ()  scalar

        # Return as a dict. The keys MUST match what the model's forward()
        # method expects. GPT2ForSequenceClassification accepts:
        #   input_ids, attention_mask, labels  (exactly these names)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         label
        }


# ─────────────────────────────────────────────────────────────
#  Data loading pipeline
# ─────────────────────────────────────────────────────────────

def load_data(
    config: DataConfig,
    tokenizer: GPT2Tokenizer
) -> Tuple["SpamDataset", "SpamDataset"]:
    """
    Runs the full data pipeline: read CSV → balance → split → wrap in Datasets.

    Args:
        config:    DataConfig with paths, column names, split ratio etc.
        tokenizer: The GPT-2 tokenizer to embed in the Dataset objects.

    Returns:
        (train_dataset, val_dataset) — two SpamDataset objects, ready for Trainer.
    """

    # ── STEP 1: Load the CSV into a DataFrame ─────────────────────────────────
    # pd.read_csv reads the file at config.dataset_path.
    # The result is a DataFrame: an in-memory table with named columns.
    # Our CSV has columns: "label" (string), "text" (string), "label_num" (int).
    print(f"Loading dataset from: {config.dataset_path}")
    df = pd.read_csv(config.dataset_path)
    print(f"  Raw size: {len(df):,} rows")
    # value_counts() tallies how many rows have each unique value in that column.
    print(f"  Class distribution:\n{df[config.label_column].value_counts().to_string()}")

    # ── STEP 2: Balance the classes ───────────────────────────────────────────
    # The raw dataset may be imbalanced.
    #
    # WHY balance matters:
    #   Imagine 90% ham / 10% spam. A "dumb" model that always outputs "ham"
    #   gets 90% accuracy — but it never catches any spam!
    #   CrossEntropyLoss optimises for accuracy, so it would happily train
    #   toward this useless solution on imbalanced data.
    #
    # Fix: undersample the majority class so both classes have equal counts.
    # We lose some majority-class data, but gain a much fairer training signal.

    # Boolean indexing: df[condition] returns only rows where condition is True.
    spam_df = df[df[config.label_column] == 1]  # All spam rows
    ham_df  = df[df[config.label_column] == 0]  # All ham rows

    # Determine how many examples to use per class.
    # min() ensures we don't try to sample more than what exists.
    n = min(len(spam_df), len(ham_df), config.max_samples_per_class)
    print(f"\n  Sampling {n} examples per class (total: {2*n})")

    # .sample(n, random_state=seed) randomly picks n rows without replacement.
    # random_state makes this deterministic — same seed = same n rows.
    spam_sampled = spam_df.sample(n, random_state=config.random_seed)
    ham_sampled  = ham_df.sample(n,  random_state=config.random_seed)

    # pd.concat stacks two DataFrames vertically (row-wise).
    # .sample(frac=1) shuffles ALL rows (frac=1 = 100% of rows, just reordered).
    # .reset_index(drop=True) replaces the original row indices (which would be
    # mixed up after concat) with a clean 0, 1, 2, ... sequence.
    df_balanced = (
        pd.concat([spam_sampled, ham_sampled])
        .sample(frac=1, random_state=config.random_seed)
        .reset_index(drop=True)
    )
    print(f"  Balanced + shuffled: {len(df_balanced)} rows")

    # ── STEP 3: Extract texts and labels as Python lists ──────────────────────
    # .tolist() converts a pandas Series (column) into a plain Python list.
    # SpamDataset expects lists, not pandas Series.
    texts  = df_balanced[config.text_column].tolist()   # List of strings
    labels = df_balanced[config.label_column].tolist()  # List of ints (0 or 1)

    # ── STEP 4: Train / validation split ──────────────────────────────────────
    # train_test_split from scikit-learn randomly partitions the data.
    #
    # test_size: fraction that goes to the "test" set (our validation set).
    #   1 - 0.8 = 0.2 → 20% validation, 80% train.
    #
    # stratify=labels: ensures both splits have the SAME class ratio.
    #   Without stratify, random chance could put 55% spam in train and
    #   45% in val, making the comparison unfair.
    #   With stratify, both train and val are guaranteed 50% spam / 50% ham.
    #
    # random_state: fixes the split for reproducibility.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=1.0 - config.train_ratio,
        random_state=config.random_seed,
        stratify=labels
    )

    print(f"  Train: {len(train_texts)} examples | Val: {len(val_texts)} examples")

    # ── STEP 5: Wrap in SpamDataset objects ───────────────────────────────────
    # SpamDataset does NOT tokenize everything now. It stores the raw texts
    # and tokenizes lazily in __getitem__ as the Trainer requests each batch.
    # This approach keeps memory usage proportional to batch_size, not dataset_size.
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset   = SpamDataset(val_texts,   val_labels,   tokenizer, config.max_length)

    print(f"\n  SpamDataset created successfully.")
    print(f"  Each call to dataset[i] returns: input_ids (128,), attention_mask (128,), labels ()")
    return train_dataset, val_dataset
