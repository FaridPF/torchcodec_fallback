# TorchCodec Fallback for TorchAudio

A lightweight fallback implementation that replaces the original **`torchcodec`** dependency in TorchAudio with a simple wrapper using **`torchaudio`** and **`soundfile`**.

> ⚠️ Use at your own risk — this is a quick workaround, not an official patch.

---

## 🧩 Overview

This module provides drop-in replacements for the missing `torchcodec` integration:
- **`load_with_torchcodec()`** — loads audio using `torchaudio` or `soundfile` if the former fails  
- **`save_with_torchcodec()`** — saves audio using `torchaudio`, or falls back to `soundfile`  

It allows you to **bypass the torchcodec dependency entirely** while maintaining core functionality for loading and saving audio tensors.

---

## 🚀 Installation

Make sure you have the dependencies installed:

```bash
pip install torch torchaudio soundfile

Then place this file (for example, _torchcodec_fallback.py) somewhere importable in your project, or directly replace the _torchcodec.py file inside your TorchAudio wrapper.
