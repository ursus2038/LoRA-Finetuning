# LoRA Fine-Tuning & Few-Shot Prompting on GSM8K

An end-to-end study of how to improve mathematical reasoning in a small language model (Qwen2.5-1.5B-Instruct) using LoRA supervised fine-tuning, few-shot prompting, and LoRA rank scaling. Evaluated on the GSM8K grade-school math benchmark.

## Overview

| Task | Method | Accuracy |
|---|---|---|
| Baseline | Zero-shot, no fine-tuning | 40% |
| SFT 1k | LoRA fine-tuning, 1,000 examples | 42% |
| SFT 3k | LoRA fine-tuning, 3,000 examples | 43% |
| SFT 3k + 3-shot | LoRA + few-shot prompting | 48% |
| SFT 3k (r=16) + 3-shot | Higher LoRA rank | 55% |
| SFT 3k (r=32) + 3-shot | Higher LoRA rank | 56% |

## Tasks

### Task 1 - Zero-Shot Baseline

Evaluates Qwen2.5-1.5B-Instruct on 100 GSM8K test questions with no fine-tuning. Answers are extracted from a `\boxed{}` marker in the model's output. Failure cases are analyzed to identify recurring error types (arithmetic errors, percentage misinterpretation, multi-step reasoning failures).

### Task 2 - LoRA Fine-Tuning with Scaling

Applies LoRA SFT using the GSM8K training split at 1k and 3k examples. Each training example is formatted as a chat-style interaction with the ground-truth solution reformatted to include a `\boxed{}` final answer. Completion-only loss is used so gradients flow only through the assistant response tokens. Accuracy is plotted as a function of training set size to study scaling behavior.

Default LoRA configuration:

| Hyperparameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q, k, v, o projections |
| Learning rate | 2e-4 (cosine schedule) |
| Effective batch size | 32 (8 per device x 4 accumulation steps) |
| Epochs | 1 |
| Max sequence length | 1,024 |

### Task 3 - Few-Shot Prompting

Evaluates k=3 shot prompting on both the base model and the best SFT model. Three demonstration examples from the GSM8K training split are prepended as user/assistant turns before each test question. Results show that few-shot prompting harms the base model (distracts from reasoning) but provides a meaningful boost to the SFT model, which has already learned the expected format and reasoning style.

### Task 4 - Data Quality Analysis

Examines the bottleneck in performance after scaling. Qualitative analysis of persistent failure modes (arithmetic errors, percentage interpretation, multi-step planning) motivates a shift from data quantity to data quality as the primary lever for further improvement.

### Task 5 - LoRA Rank Scaling

Tests the hypothesis that increasing LoRA rank allows the model to learn more complex reasoning patterns. Trains 3k-example SFT models at r=16 and r=32 and evaluates each with 3-shot prompting. Results show consistent gains: r=8 -> 48%, r=16 -> 55%, r=32 -> 56%, with diminishing returns at higher rank.

## Setup

**Prerequisites:** Python 3.10+, a CUDA-capable GPU (16GB+ VRAM recommended), and a Hugging Face account with access to `Qwen/Qwen2.5-1.5B-Instruct`.

Install dependencies:

```bash
pip install trl peft accelerate -q
```

Set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Usage

Open `lora_finetuning_gsm8k.ipynb` and run cells in order. Each task section is self-contained.

**Task 1:** Run the baseline evaluation cells. The model is loaded, evaluated on 100 GSM8K test questions, and cleaned up from GPU memory.

**Task 2:** Run the training cells for 1k and 3k examples. Each saves a LoRA adapter to disk (`1kpath/final_adapter`, `3kpath/final_adapter`), then loads and evaluates it. The scaling plot is generated automatically.

**Task 3:** Run the few-shot setup cell to build demonstration examples, then the evaluation cells for both the base model and SFT model.

**Task 5:** Run the r=16 and r=32 training cells sequentially. Each saves its adapter, evaluates with 3-shot prompting, and the final cell plots accuracy vs. LoRA rank.

## Configuration

Key parameters are defined near the top of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model from Hugging Face |
| `GSM8K_TRAIN_SAMPLES` | 1000 / 3000 | Training examples per run |
| `num_samples` | 100 | GSM8K test questions evaluated |
| `batch_size` | 16 | Inference batch size |

## Output

- `1kpath/final_adapter/` - LoRA adapter trained on 1,000 examples (r=8)
- `3kpath/final_adapter/` - LoRA adapter trained on 3,000 examples (r=8)
- `3k_r16_path/final_adapter/` - LoRA adapter trained on 3,000 examples (r=16)
- `3k_r32_path/final_adapter/` - LoRA adapter trained on 3,000 examples (r=32)
- Accuracy vs. training examples plot
- Accuracy vs. LoRA rank plot

## Acknowledgments

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Grade school math benchmark (Cobbe et al., 2021)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) - Base model
- [PEFT](https://github.com/huggingface/peft) - LoRA implementation via Hugging Face
- [TRL](https://github.com/huggingface/trl) - SFTTrainer for supervised fine-tuning
