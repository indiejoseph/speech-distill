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

### Quick Start

Basic training with audio-to-speech token processing:

```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./student_model_aligned \
    --dataset_path /path/to/your/dataset \
    --use_processor \
    --teacher_prefix "<|task_podcast|><|SPEAKER_0|>" \
    --student_prefix "" \
    --text_bos "<|text_start|>" \
    --text_eos "<|text_end|>" \
    --speech_bos "<|semantic_token_start|>" \
    --speech_eos "<|semantic_token_end|>" \
    --max_length 512 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --temperature 2.0 \
    --alpha 0.5
```

### Training Parameters

**Model Arguments:**
- `--teacher_model`: Path or HuggingFace ID of teacher model (default: `Soul-AILab/SoulX-Podcast-1.7B-dialect`)
- `--student_model`: Path or HuggingFace ID of student model (default: `./student_model_aligned`)

**Dataset Arguments:**
- `--dataset_path`: Path to dataset (required)
- `--max_length`: Maximum sequence length (default: 512)
- `--use_processor`: Enable audio-to-speech token conversion (recommended)

**Prefix Configuration:**

The system supports three types of prefixes:

1. **String prefix** (simple, fixed for all examples):
   ```bash
   --student_prefix "<|task_podcast|><|SPEAKER_0|>"
   ```

2. **Dict prefix** (language-dependent mapping):
   ```bash
   --student_prefix '{"en": "[EN]", "zh": "[ZH]", "yue": "[YUE]", "default": ""}'
   ```

3. **Function prefix** (dynamic, see [PREFIX_GUIDE.md](PREFIX_GUIDE.md)):
   - Modify `train.py` to import custom functions from `prefix_functions.py`
   - Allows prefix generation based on both text content and language

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
├── train.py                    # Main training script
├── data.py                     # Dataset processor and collators
├── distillation_loss.py        # KL divergence distillation loss
├── prepare_student.py          # Student vocab alignment
├── utils.py                    # Audio processing utilities
└── requirements.txt           # Python dependencies
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

## Advanced Usage

### Custom Prefix Functions
See [PREFIX_GUIDE.md](PREFIX_GUIDE.md) for detailed guide on creating dynamic prefix functions.

Example:
```python
def custom_prefix(text: str, lang: str) -> str:
    """Generate prefix based on text and language."""
    task = "<|task_podcast|>"
    speaker = "<|SPEAKER_0|>"
    lang_tag = f"[{lang.upper()}]"
    
    # Dynamic logic
    if len(text) > 100:
        return f"{task}{speaker}{lang_tag}[LONG]"
    
    return f"{task}{speaker}{lang_tag}"
```

### Legacy Mode (Pre-computed Speech Tokens)
If your dataset already has `speech_tokens` computed, omit `--use_processor`:
```bash
python train.py \
    --teacher_model Soul-AILab/SoulX-Podcast-1.7B-dialect \
    --student_model ./student_model_aligned \
    --dataset_path /path/to/dataset/with/speech_tokens \
    --teacher_prefix "" \
    --student_prefix ""
```

## References

- Teacher Model: [Soul-AILab/SoulX-Podcast-1.7B-dialect](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect)
- Student Base: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Speech Tokenizer: s3tokenizer
