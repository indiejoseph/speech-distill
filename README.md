# Speech Distillation Project

This project aims to distill the `Soul-AILab/SoulX-Podcast-1.7B-dialect` model into `Qwen/Qwen3-0.6B`.

## Models
- **Teacher**: `Soul-AILab/SoulX-Podcast-1.7B-dialect`
- **Student**: `Qwen/Qwen3-0.6B`

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Prepare Student Model
Align student vocabulary with teacher (expand vocab to include speech tokens):
```bash
python prepare_student.py
```

This creates `./student_model_aligned` with expanded vocabulary.

## Training

### Stage 1: Text-to-Speech Token Alignment (Optional Warm-up)

Before running the main distillation training, you can optionally run Stage 1 to help the model familiarize itself with the newly added speech tokens. This stage fine-tunes only the new speech token embeddings while keeping all other model weights frozen.

**Benefits:**
- Helps the model quickly learn the mapping between text and new speech tokens
- Prevents catastrophic forgetting of pre-trained weights
- Acts as a warm-up before full distillation training
- Efficient training with minimal memory footprint

**Usage:**
```bash
python stage1.py \
    --model_path ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/dataset \
    --output_dir ./stage1_output \
    --num_new_tokens 1000 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_checkpointing
```

**Stage 1 Parameters:**

**Model & Data Arguments:**
- `--model_path`: Path to the model with expanded vocabulary (required)
- `--dataset_path`: Path to training dataset (required)
- `--output_dir`: Output directory for checkpoints and final model (required)
- `--num_new_tokens`: Number of newly added speech tokens (default: 1000)

**Training Parameters:**
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 4)
- `--eval_batch_size`: Batch size for evaluation (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Number of warmup steps (default: 1000)
- `--weight_decay`: Weight decay (default: 0.01)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--logging_steps`: Logging frequency (default: 50)
- `--save_steps`: Save checkpoint frequency (default: 500)
- `--eval_steps`: Evaluation frequency (default: 500)
- `--eval_size`: Evaluation set size as fraction (default: 0.1)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--num_workers`: Data loading workers (default: 4)

**Token Configuration:**
- `--text_bos`: Text begin-of-sequence token (default: `<|text_start|>`)
- `--text_eos`: Text end-of-sequence token (default: `<|text_end|>`)
- `--speech_bos`: Speech begin-of-sequence token (default: `<|semantic_token_start|>`)
- `--speech_eos`: Speech end-of-sequence token (default: `<|semantic_token_end|>`)
- `--prefix`: Prefix to add before text (default: "")
- `--text_prefix`: Text prefix after eos, before speech tokens (default: "")

**Optimization Options:**
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory (enabled by default)
- `--use_8bit_optimizer`: Use 8-bit AdamW optimizer (enabled by default)
- `--use_wandb`: Log to Weights & Biases
- `--seed`: Random seed (default: 42)

**How It Works:**
1. Loads the model with expanded vocabulary (including new speech tokens)
2. Freezes all model weights except the new speech token embeddings
3. Uses gradient masking to prevent updates to original vocabulary tokens
4. Trains with SFTTrainer on text-to-speech token alignment task
5. Saves the warm-up checkpoint for use in Stage 2

**Stage 1 Output:**
- `./stage1_output/checkpoint-*/`: Training checkpoints
- `./stage1_output/final_model/`: Final warm-up model
- Best model loaded automatically if eval dataset is used

**Example: Multi-epoch Stage 1 Training**
```bash
python stage1.py \
    --model_path ./pretrained_models/Qwen3-0.6B \
    --dataset_path /notebooks/bert-vits2/frankenstein-matcha2/tmp/dataset_small \
    --output_dir ./stage1_output \
    --num_new_tokens 1000 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --eval_size 0.1 \
    --gradient_checkpointing \
    --use_wandb
```

### Stage 1.5: Dataset Preparation (Optional)

Pre-process your dataset once to avoid on-the-fly audio-to-speech token conversion during training. This significantly speeds up training iterations:

```bash
python prepare_dataset.py \
    --dataset_path /path/to/raw/dataset \
    --output_path ./processed_dataset \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --max_length 512 \
    --num_proc 4
```

**Benefits:**
- One-time preprocessing overhead, then instant training iterations
- Avoids repeated audio-to-speech token computation during training
- Results in dataset with `student_input_ids`, `teacher_input_ids`, etc.

**Output:** A dataset with pre-computed student and teacher input sequences ready for training.

### Stage 2: Knowledge Distillation (Main Training)

**Option A: Training with pre-processed dataset (recommended)**

```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path ./processed_dataset \
    --teacher_prefix "<|task_podcast|><|SPEAKER_0|>" \
    --student_prefix "" \
    --max_length 512 \
    --use_lora \
    --temperature 2.0 \
    --alpha 0.5
```

**Option B: Training with on-the-fly processing (slower but more flexible)**

```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/raw/dataset \
    --teacher_prefix "<|task_podcast|><|SPEAKER_0|>" \
    --student_prefix "" \
    --text_bos "<|text_start|>" \
    --text_eos "<|text_end|>" \
    --speech_bos "<|semantic_token_start|>" \
    --speech_eos "<|semantic_token_end|>" \
    --max_length 512 \
    --use_lora \
    --temperature 2.0 \
    --alpha 0.5
```

The script automatically detects whether the dataset is pre-processed and applies the appropriate data pipeline.

### Training Parameters

**Model Arguments:**
- `--teacher_model`: Path or HuggingFace ID of teacher model (default: `Soul-AILab/SoulX-Podcast-1.7B-dialect`)
- `--student_model`: Path or HuggingFace ID of student model (default: `./student_model_aligned`)
- `--load_teacher_in_8bit`: Load teacher in 8-bit quantization (~75% VRAM reduction, disables sparse distillation)
- `--load_teacher_in_4bit`: Load teacher in 4-bit quantization (~80% VRAM reduction, disables sparse distillation)

**Dataset Arguments:**
- `--dataset_path`: Path to dataset (required) - can be raw or pre-processed
- `--max_length`: Maximum sequence length (default: 512)
- `--test_size`: Evaluation set size (default: 10)

**Prefix Configuration:**

The system supports two types of prefixes:

1. **String prefix** (simple, fixed for all examples):
   ```bash
   --student_prefix "<|task_podcast|><|SPEAKER_0|>"
   ```

2. **Dict prefix** (language-dependent mapping):
   ```bash
   --student_prefix '{"en": "[EN]", "zh": "[ZH]", "yue": "[YUE]", "default": ""}'
   ```

**Token Delimiters:**
- `--text_bos`: Text begin-of-sequence token (default: `<|text_start|>`)
- `--text_eos`: Text end-of-sequence token (default: `<|text_end|>`)
- `--speech_bos`: Speech begin-of-sequence token (default: `<|semantic_token_start|>`)
- `--speech_eos`: Speech end-of-sequence token (default: `<|semantic_token_end|>`)

**LoRA Arguments:**
- `--use_lora`: Enable LoRA for efficient training
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)

**Distillation Arguments:**
- `--temperature`: Distillation temperature for softening logits (default: 2.0)
- `--alpha`: Weight for task loss vs distillation loss (default: 0.5)
  - `loss = alpha * task_loss + (1 - alpha) * distillation_loss`
- `--top_k`: Number of top logits to keep for sparse distillation (default: 128, only used with on-the-fly extraction)

### Dataset Requirements

When using `--use_processor`, your dataset should have:
- `audio`: Audio file path or HuggingFace audio dict
- `text`: Transcription text
- `lang` (optional): Language code (e.g., "en", "zh", "yue")

Example dataset structure:
```python
{
    "audio": "path/to/audio.wav",  # or HF audio dict
    "text": "到咗落車嘅時候，連媽媽都走失埋。",
    "lang": "yue"
}
```

### Training Examples

**Example 1: Basic training with LoRA**
```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/dataset \
    --use_processor \
    --use_lora \
    --teacher_prefix "<|task_podcast|><|SPEAKER_0|>" \
    --student_prefix ""
```

**Example 2: Multi-language with per-language prefixes**
```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/multilingual/dataset \
    --use_processor \
    --use_lora \
    --teacher_prefix '{"en": "<|task_podcast|><|SPEAKER_0|>[EN]", "zh": "<|task_podcast|><|SPEAKER_0|>[ZH]", "yue": "<|task_podcast|><|SPEAKER_0|>[YUE]", "default": "<|task_podcast|><|SPEAKER_0|>"}' \
    --student_prefix '{"en": "[EN] ", "zh": "[中文] ", "yue": "[粵語] ", "default": ""}'
```

**Example 3: Custom distillation settings**
```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/dataset \
    --use_processor \
    --use_lora \
    --temperature 3.0 \
    --alpha 0.3 \
    --lora_r 16 \
    --lora_alpha 32
```

## Project Structure

```
.
├── train.py                    # Stage 2: Main knowledge distillation training
├── prepare_dataset.py          # Pre-process raw dataset (audio → tokenized inputs)
├── extract_teacher_logits.py   # Extract sparse top-K teacher logits for efficient distillation
├── stage1.py                   # Stage 1: Text-to-speech token alignment warm-up
├── data.py                     # Dataset processor and collators
├── distillation_loss.py        # KL divergence distillation loss
├── prepare_student.py          # Student vocab alignment
├── utils.py                    # Audio processing utilities
└── requirements.txt            # Python dependencies
```

## Components

### Dataset Processor
`SpeechDistillDatasetProcessor` handles:
1. Converting audio files to speech tokens using `s3tokenizer`
2. Formatting text + speech tokens with configurable delimiters
3. Tokenizing for model input
4. Supporting dynamic prefix generation based on text and language

### Collators
- `ProcessedDataCollator`: For data processed by `SpeechDistillDatasetProcessor`
- `SpeechDistillDataCollator`: Legacy collator for pre-computed speech tokens

### Distillation Loss
Combines:
- **Task Loss**: Standard cross-entropy on target tokens
- **Distillation Loss**: KL divergence between teacher and student logits (softened by temperature)

Formula: `loss = α × task_loss + (1 - α) × distillation_loss`

## Sparse Distillation

### How It Works

**Sparse distillation** is a memory optimization technique that reduces VRAM usage during training by **50-80%** while maintaining distillation quality. Instead of computing KL divergence on all vocabulary tokens (typically 150K+), we only compute it on the **Top-K most likely tokens** from the teacher model.

#### Dense Distillation (Standard)
```
Teacher logits: [150K vocab] 
Student logits: [150K vocab]
KL divergence computed on: All 150K tokens
Memory: High (~GB per batch)
```

#### Sparse Distillation (Optimized)
```
Teacher Top-K: [128 values, 128 indices]
Student logits: [150K vocab]
KL divergence computed on: Only 128 teacher's top tokens
Memory: Low (~MB per batch) ✓
```

### Two Modes

#### 1. **Pre-extracted Sparse Logits** (Fastest)
Pre-compute and cache teacher's top-K logits using `extract_teacher_logits.py`:

```bash
python extract_teacher_logits.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/raw/dataset \
    --output_path ./dataset_with_logits \
    --top_k 128 \
    --batch_size 4
```

**Benefits:**
- Zero overhead during training (logits already computed)
- Fastest training iterations
- Exact same results (deterministic extraction)

**Output:** Dataset with `teacher_top_k_v` and `teacher_top_k_i` columns

Then use in training:
```bash
python train.py \
    --dataset_path ./dataset_with_logits \
    --student_model ./pretrained_models/Qwen3-0.6B \
    ...
```

#### 2. **On-the-fly Sparse Extraction** (Flexible)
Compute sparse logits during training automatically:

```bash
python train.py \
    --dataset_path /path/to/raw/dataset \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --top_k 128 \
    ...
```

**Benefits:**
- No preprocessing step required
- Flexible (can change `--top_k` between runs)
- Works with raw datasets

**Trade-off:** ~10-20% VRAM overhead per batch (teacher forward pass required)

### VRAM Comparison

| Mode | VRAM | Speed | Notes |
|------|------|-------|-------|
| Dense (full vocab) | 100% baseline | Slow | All 150K tokens |
| Sparse (pre-extracted) | ~20-30% | Fast ✓ | No training overhead |
| Sparse (on-the-fly) | ~40-50% | Medium | Slight slowdown/batch |
| Dense (8-bit teacher) | ~30% | Slow | Quantization + full vocab |
| Sparse (4-bit teacher) | Not available | - | Falls back to dense |

### When to Use Each Mode

**Use Pre-extracted** if:
- Training the same dataset multiple times
- Want maximum speed and minimal VRAM
- Have storage for enriched dataset

**Use On-the-fly** if:
- One-time training
- Experimenting with different top_k values
- Storage is limited

**Use Dense (with quantization)** if:
- Extremely tight VRAM constraints (~2GB teacher)
- Quality is less critical than memory savings
- Use `--load_teacher_in_8bit` or `--load_teacher_in_4bit`

### Implementation Details

**Sparse KL Divergence:**
```python
# Get teacher's top-K indices for this batch
# teacher_top_k_i: [batch, seq_len, K]  <- indices
# teacher_top_k_v: [batch, seq_len, K]  <- log probs

# Gather student logits at those indices
student_topk = torch.gather(
    student_logits, 
    dim=-1, 
    index=teacher_top_k_i
)  # [batch, seq_len, K]

# KL divergence on only K tokens
# loss = sum(exp(teacher_topk) * (teacher_topk - student_topk))
```

**Memory Optimization:**
- Teacher logits: 150K float32 → 128 float16 + 128 int32 = **99.7% reduction**
- With batch size 4, seq_len 512: ~2GB → ~20MB for sparse logits

## Advanced Features

### Extract Teacher Logits Script

Pre-compute teacher's top-K logprobs for efficient training:

```bash
python extract_teacher_logits.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --dataset_path /path/to/raw/dataset \
    --output_path ./dataset_enriched \
    --top_k 128 \
    --batch_size 4 \
    --teacher_prefix "<|task_podcast|><|SPEAKER_0|>" \
    --student_prefix ""
```

**Parameters:**
- `--teacher_model`: Teacher model to extract from
- `--student_model`: Student model (used for tokenizer)
- `--dataset_path`: Raw dataset path (required)
- `--output_path`: Output enriched dataset path (required)
- `--top_k`: Number of top logits per token (default: 128)
- `--batch_size`: Extraction batch size (default: 4)
- `--teacher_prefix`, `--student_prefix`: Same as training
- `--num_proc`: Parallel processing workers

**Output:** Dataset with added columns:
- `teacher_top_k_v`: Top-K log probabilities [seq_len, K] (fp16)
- `teacher_top_k_i`: Top-K token indices [seq_len, K] (int32)

### Prepare Dataset Script

Pre-process raw dataset for faster training:

```bash
python prepare_dataset.py \
    --dataset_path /path/to/raw/dataset \
    --output_path ./processed_dataset \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --max_length 512 \
    --num_proc 4 \
    --device cuda
```

**Parameters:**
- `--dataset_path`: Raw dataset path (required)
- `--output_path`: Output processed dataset path (required)
- `--student_model`: Student model for tokenizer (default: `./pretrained_models/Qwen3-0.6B`)
- `--max_length`: Max sequence length (default: 512)
- `--teacher_prefix`, `--student_prefix`: Prefix configuration
- `--num_proc`: Parallel workers (use >1 carefully with GPU)
- `--device`: Device for processing ('cuda' or 'cpu')

**Output:** Dataset with pre-computed sequences:
- `student_input_ids`: Tokenized student input
- `student_attention_mask`: Attention mask for student
- `teacher_input_ids`: Tokenized teacher input
- `teacher_attention_mask`: Attention mask for teacher

**Workflow:**
1. Run `prepare_dataset.py` once → creates `processed_dataset`
2. Run `extract_teacher_logits.py` on `processed_dataset` → adds sparse logits
3. Train with `train.py` using the enriched dataset → fast, zero overhead

### Recommended Pipeline

For **maximum efficiency and speed**:
```bash
# Step 1: Preprocess dataset (one-time)
python prepare_dataset.py \
    --dataset_path /path/to/raw/data \
    --output_path ./processed_data

# Step 2: Extract sparse teacher logits (one-time)
python extract_teacher_logits.py \
    --dataset_path ./processed_data \
    --output_path ./processed_data_with_logits

# Step 3: Train (fast iterations)
python train.py \
    --dataset_path ./processed_data_with_logits \
    --student_model ./pretrained_models/Qwen3-0.6B \
    --use_lora
```

**Time Investment:**
- Preprocessing: ~5-30 min (one-time, depends on dataset size)
- Logit extraction: ~10-60 min (one-time, depends on model size + data)
- Training: Fast iteration (precomputed data = instant batches)

**VRAM During Training:**
- Student: ~2GB
- Teacher (sparse path): ~1GB
- Total: ~3GB (vs 8-10GB for dense path)

## Advanced Usage

### Memory-Constrained Training

If you have limited VRAM:

**Option 1: Full Sparse Distillation (~3GB)**
```bash
# Pre-extract logits once
python extract_teacher_logits.py \
    --dataset_path /path/to/raw/data \
    --output_path ./data_with_logits

# Train with sparse distillation
python train.py \
    --dataset_path ./data_with_logits \
    --use_lora \
    --gradient_checkpointing
```

**Option 2: Quantized Teacher (~2.5GB)**
```bash
python train.py \
    --dataset_path /path/to/data \
    --load_teacher_in_8bit \
    --use_lora \
    --gradient_checkpointing
```

**Option 3: Aggressive Quantization (~2GB)**
```bash
python train.py \
    --dataset_path /path/to/data \
    --load_teacher_in_4bit \
    --use_lora \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

## Troubleshooting

### Flash Attention Installation Issues

If you encounter issues with flash-attention, you can install a pre-built wheel:

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-2.7.4+cu128torch2.7-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

This installs `flash_attn==2.7.4+cu128torch2.7` which is compatible with CUDA 12.8 and PyTorch 2.7.

## References

- Teacher Model: [Soul-AILab/SoulX-Podcast-1.7B-dialect](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect)
- Student Base: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Speech Tokenizer: s3tokenizer
