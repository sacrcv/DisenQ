"""DisenQ: Disentangling Q-Former for Activity-Biometrics (ICCV 2025).

Official implementation of https://arxiv.org/abs/2507.07262, built on the
BLIP-2 Q-Former from Salesforce LAVIS.

    Azad & Rawat. DisenQ: Disentangling Q-Former for Activity-Biometrics. ICCV 2025.
"""

import os

# The frozen text encoder tokenises inside the main process, which makes the
# HuggingFace fast tokenizer complain loudly once the dataloader workers fork.
# Tokenisation here is tiny (three short prompts per clip), so parallelism buys
# nothing and silencing it keeps the logs readable.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
