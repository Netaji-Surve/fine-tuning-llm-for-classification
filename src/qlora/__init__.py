"""
src/qlora/__init__.py

This file marks the `qlora` directory as a Python *package*, which means
Python treats it as a module you can import from.

Without this file, `from src.qlora.config import LoRAConfig` would fail
with a ModuleNotFoundError, even if config.py exists.

The file can be empty (and often is), but we use it to re-export the most
commonly used symbols so callers can write:

    from src.qlora import LoRAConfig, train          ← short form
    from src.qlora.config import LoRAConfig          ← explicit form (also fine)

Both forms work; the short form is just more convenient in notebooks.
"""

# Re-export the public API so users of this package can do one-line imports.
from .config  import LoRAConfig, TrainingConfig, DataConfig
from .model   import create_tokenizer, create_model, print_trainable_parameters
from .dataset import SpamDataset, load_data
from .train   import train
