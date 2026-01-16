# GAD Evaluation Guide

This guide explains how to evaluate models trained with Generative Adversarial Distillation (GAD) using LLM-as-a-Judge methodology.

## Overview

The GAD evaluation pipeline consists of two main steps:

1. **Generation Phase** - Generate responses from trained model checkpoints
2. **Scoring Phase** - Compare student vs teacher responses using GPT-4o as judge

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GAD Evaluation Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. GENERATION                                                           │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  For each checkpoint:                                         │    │
│     │    • Convert FSDP shards to HuggingFace format               │    │
│     │    • Generate responses on validation set                     │    │
│     │    • Output: {val_data}_generation_results.jsonl             │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                               ↓                                          │
│  2. PAIRWISE COMPARISON (LLM-as-a-Judge)                                │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  For each query:                                              │    │
│     │    • Student response vs Teacher response                     │    │
│     │    • GPT-4o judges which is better                           │    │
│     │    • Position shuffling to mitigate bias                     │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                               ↓                                          │
│  3. METRICS                                                              │
│     • Student win rate vs teacher (target: ~50%)                        │
│     • Invalid judgment rate                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Environment Setup

```bash
# Install required packages
pip install openai tqdm

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Data Requirements

- **Trained model checkpoints**: Located at `/tmp/{exp_name}/global_step_{ckpt}/`
- **Teacher responses**: JSONL file with teacher model outputs on the same queries
- **Validation dataset**: Parquet file (e.g., `lmsys_gpt5_chat_filtered_test.parquet`)

## Quick Start

### Option 1: Full Pipeline

```bash
cd gad

# Run complete evaluation pipeline
bash scripts/eval/run_evaluation.sh \
    --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
    --ckpt 1000 \
    --teacher_file /path/to/teacher_responses.jsonl \
    --judge_model gpt-4o
```

### Option 2: Step-by-Step

#### Step 1: Generate Student Responses

```bash
cd gad

# Switch to eval branch (if needed)
cd verl && git checkout eval && cd ..

# Generate responses for a specific checkpoint
bash scripts/generate/generate.sh \
    --model /tmp/Qwen2.5-7B-Instruct \
    --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
    --val_data lmsys \
    --ckpt_start 1000 \
    --ckpt_end 1000 \
    --ckpt_step 1 \
    --nnodes 1 \
    --ngpus 8
```

#### Step 2: Run LLM-as-a-Judge Evaluation

```bash
cd gad

python scripts/eval/evaluate_pairwise.py \
    --student /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/global_step_1000/lmsys_generation_results.jsonl \
    --teacher /path/to/teacher_responses.jsonl \
    --output /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/global_step_1000/eval_results.json \
    --judge-model gpt-4o \
    --workers 4
```

## Detailed Usage

### evaluate_pairwise.py

The main evaluation script that performs pairwise comparisons using LLM-as-a-Judge.

```
usage: evaluate_pairwise.py [-h] --student STUDENT --teacher TEACHER --output OUTPUT
                            [--judge-model MODEL] [--workers N] [--limit N]
                            [--no-shuffle] [--seed SEED] [--resume]

Arguments:
  --student PATH      Path to student generation results (JSONL or Parquet)
  --teacher PATH      Path to teacher generation results (JSONL or Parquet)
  --output PATH       Output file for evaluation results (JSON)
  --judge-model       Model to use as judge (default: gpt-4o)
  --workers N         Number of parallel API workers (default: 4)
  --limit N           Limit comparisons for testing
  --no-shuffle        Disable position shuffling (not recommended)
  --seed N            Random seed for reproducibility (default: 42)
  --resume            Resume from existing output file
```

### run_evaluation.sh

Convenience wrapper script that handles the full pipeline.

```
usage: run_evaluation.sh [OPTIONS]

Options:
  --exp_name NAME      Experiment name (required)
  --ckpt NUMBER        Checkpoint number to evaluate (required)
  --judge_model MODEL  Judge model (default: gpt-4o)
  --workers N          Number of parallel workers (default: 4)
  --val_data NAME      Validation dataset name (default: lmsys)
  --teacher_file PATH  Path to teacher responses JSONL
  --limit N            Limit number of comparisons
  --resume             Resume from existing output file
  --skip_generation    Skip generation step
```

## Input/Output Formats

### Input: Generation Results (JSONL or Parquet)

The script supports both JSONL and Parquet formats (auto-detected by file extension).

**JSONL format** - Each line should be a JSON object:
```json
{"prompt": "User query here...", "response": "Model response here..."}
```

**Parquet format (recommended)** - Columnar format with columns like:
- `content` - Query/prompt (can be a conversation list)
- `teacher_response` - Model response

Alternative key names are supported: `query`, `question`, `content` for prompts; `output`, `answer`, `completion`, `teacher_response` for responses.

**Conversation format**: If `content` is a list of messages (e.g., `[{"role": "user", "content": "..."}]`), user messages are automatically extracted.

### Output: Evaluation Results (JSON)

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00",
    "total_comparisons": 600,
    "valid_comparisons": 595,
    "invalid_comparisons": 5
  },
  "stats": {
    "student_wins": 298,
    "teacher_wins": 297,
    "invalid": 5,
    "total": 600
  },
  "win_rate": 0.5008,
  "results": [
    {
      "query": "What is machine learning?",
      "student_response": "Machine learning is...",
      "teacher_response": "Machine learning refers to...",
      "winner": "student",
      "swapped": false,
      "raw_response": "\\boxed{Assistant 1}",
      "valid": true
    }
  ]
}
```

## Interpreting Results

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Win Rate** | Percentage of comparisons where student wins | ~50% (matches teacher) |
| **Invalid Rate** | Percentage of unparseable judge responses | <5% |

### What Results Mean

- **Win Rate ~50%**: Student model is comparable to teacher (successful distillation)
- **Win Rate >50%**: Student may be outperforming teacher on some aspects
- **Win Rate <40%**: Student needs more training or different hyperparameters

### Note on ROUGE-L

From the GAD paper:
> "ROUGE-L scores of GAD can be lower than those of SeqKD because ROUGE-L is a relatively local metric that primarily captures n-gram overlap rather than deeper stylistic or semantic qualities. **We observe that higher ROUGE-L scores do not necessarily correspond to better performance in either automatic or human evaluations.** Consequently, ROUGE-L is used solely as a training diagnostic."

**TL;DR**: Use LLM-as-a-Judge win rate, not ROUGE-L, for final evaluation.

## LLM-as-a-Judge Methodology

This evaluation uses the LLM-as-a-Judge methodology from [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685).

### How It Works

1. **Judge Prompt**: GPT-4o receives both responses with instructions to evaluate quality
2. **Position Shuffling**: Responses are randomly ordered to mitigate position bias
3. **Verdict Extraction**: Judge outputs `\boxed{Assistant 1}` or `\boxed{Assistant 2}`
4. **Score Aggregation**: Win rates are calculated across all comparisons

### Bias Mitigations

| Bias | Mitigation |
|------|------------|
| Position bias | Random shuffling of response order |
| Verbosity bias | Explicit instruction to ignore length |
| Self-enhancement | Using different model as judge |

### Judge Prompt (MYPROMPT2)

The evaluation uses a comprehensive judge prompt that instructs the model to:
- Prioritize instruction-following accuracy
- Consider harmlessness for harmful content
- Avoid position and length biases
- Output only `\boxed{Assistant 1}` or `\boxed{Assistant 2}`

## Parallel Evaluation of Multiple Checkpoints

To evaluate multiple checkpoints in parallel:

```bash
cd gad

# Generate responses for multiple checkpoints
bash scripts/generate/parallel_generate.sh

# Evaluate each checkpoint (can be parallelized)
for ckpt in 800 850 900 950 1000; do
    python scripts/eval/evaluate_pairwise.py \
        --student /tmp/exp_name/global_step_${ckpt}/lmsys_generation_results.jsonl \
        --teacher /path/to/teacher_responses.jsonl \
        --output /tmp/exp_name/global_step_${ckpt}/eval_results.json \
        --judge-model gpt-4o &
done
wait
```

## Cost Estimation

Using GPT-4o as judge:

| Comparisons | Estimated Cost | Time (4 workers) |
|-------------|---------------|------------------|
| 100 | ~$2-5 | ~5 min |
| 600 | ~$15-30 | ~30 min |
| 5000 | ~$100-200 | ~4 hours |

For cheaper testing, use `--judge-model gpt-4o-mini` (10x cheaper but slightly less accurate).

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: OPENAI_API_KEY environment variable not set
```
Solution: `export OPENAI_API_KEY=your_key`

**2. Rate Limiting**
```
Retry due to rate limit: 429
```
Solution: Reduce `--workers` or add backoff (handled automatically)

**3. High Invalid Rate**
```
Invalid judgments: 50/100 (50.0%)
```
Possible causes:
- API errors (check response content)
- Judge model issues (try different model)
- Malformed responses

**4. Missing Files**
```
Error: Student results file not found
```
Solution: Run generation first or check file paths

### Resuming Interrupted Evaluation

If evaluation is interrupted, use `--resume` to continue:

```bash
python scripts/eval/evaluate_pairwise.py \
    --student results.jsonl \
    --teacher teacher.jsonl \
    --output eval.json \
    --resume
```

## Files Reference

| File | Description |
|------|-------------|
| `scripts/eval/evaluate_pairwise.py` | Main LLM-as-a-Judge evaluation script |
| `scripts/eval/run_evaluation.sh` | Full pipeline wrapper script |
| `scripts/generate/generate.sh` | Response generation script |
| `scripts/generate/parallel_generate.sh` | Parallel generation for multiple checkpoints |
| `tools/merge_model2hf.py` | Convert FSDP shards to HuggingFace format |
| `deepscaler/utils.py` | OpenAI API wrapper (reused) |
| `deepscaler/rewards/judge_extractor.py` | Judge verdict extraction (reused) |

## Citation

If you use this evaluation methodology, please cite:

```bibtex
@article{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and others},
  journal={arXiv preprint arXiv:2306.05685},
  year={2023}
}

@article{gad2024,
  title={Generative Adversarial Distillation of Large Language Models},
  author={...},
  journal={arXiv preprint arXiv:2511.10643},
  year={2024}
}
```
