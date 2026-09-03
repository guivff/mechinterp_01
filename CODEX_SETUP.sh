#!/usr/bin/env bash
# Paste this as the "setup script" of the Codex environment for this repo.
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || pip install -q torch
pip install -q transformers peft trl datasets accelerate numpy scikit-learn requests pytest matplotlib
