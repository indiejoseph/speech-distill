import torch
import argparse
import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from distillation_loss import DistillationLoss
from peft import LoraConfig, get_peft_model
from data import (
    SpeechDistillDatasetProcessor,
    ProcessedDataCollator,
    DistillationDataProcessor,
    align_prefixes,
    parse_prefix,
)


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
        self.distill_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract metadata
        teacher_input_ids = inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = inputs.pop("teacher_attention_mask", None)
        speech_mask = inputs.pop("speech_token_mask", None)

        # Extract pre-calculated logits if they exist
        teacher_top_k_v = inputs.pop("teacher_top_k_v", None)
        teacher_top_k_i = inputs.pop("teacher_top_k_i", None)

        # Student forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits
        labels = inputs.get("labels")

        teacher_logits = None
        # Teacher forward pass (ONLY if pre-calculated logits are missing)
        if teacher_top_k_v is None and self.teacher_model is not None:
            with torch.no_grad():
                if teacher_input_ids is not None:
                    teacher_outputs = self.teacher_model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                    )
                else:
                    teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

        # Compute distillation loss
        loss, task_loss, distill_loss, teacher_loss = self.distill_loss_fn(
            student_logits=student_logits,
            labels=labels,
            teacher_logits=teacher_logits,
            teacher_top_k_v=teacher_top_k_v,
            teacher_top_k_i=teacher_top_k_i,
            speech_token_mask=speech_mask,
        )

        # Log individual losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "student_loss": task_loss.item(),
                    "teacher_loss": teacher_loss.item(),
                    "distill_loss": distill_loss.item(),
                }
            )

        return (loss, outputs) if return_outputs else loss


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_id = config.teacher_model
    student_path = config.student_model

    teacher_prefix = parse_prefix(config.teacher_prefix)
    student_prefix = parse_prefix(config.student_prefix)
    text_prefix = parse_prefix(config.text_prefix)

    if not os.path.exists(student_path) and not student_path.startswith("Qwen/"):
        print(
            f"Warning: Student model path {student_path} might not exist locally. Ensure it is a valid HF ID or path."
        )

    # Load models
    print(f"Loading teacher model: {teacher_id}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print(f"Loading student model: {student_path}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if config.use_lora:
        print("Applying LoRA to student model...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["embed_tokens", "lm_head"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    if config.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        student_model.gradient_checkpointing_enable()
        # Required for gradient checkpointing when using PEFT
        student_model.enable_input_require_grads()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Align teacher and student prefixes by left-padding the shorter one
    print("Aligning teacher and student prefixes...")
    teacher_prefix, student_prefix = align_prefixes(
        teacher_prefix, student_prefix, tokenizer
    )

    # Load and process dataset
    print(f"Loading dataset from: {config.dataset_path}")

    # Check if it's a Hugging Face dataset or local disk dataset
    try:
        if os.path.exists(config.dataset_path):
            raw_dataset = load_from_disk(config.dataset_path)
        else:
            raw_dataset = load_dataset(config.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Get the train split
    if isinstance(raw_dataset, dict):
        train_dataset = raw_dataset.get("train", raw_dataset)
    else:
        train_dataset = raw_dataset

    print(f"Dataset loaded: {len(train_dataset)} examples")
    print(f"Dataset columns: {train_dataset.column_names}")

    # Process dataset with SpeechDistillDatasetProcessor
    # Create separate processors for student and teacher with different prefixes
    print("Using SpeechDistillDatasetProcessor to convert audio to speech tokens...")

    # Split dataset into train and test first
    if config.test_size > 0:
        print(
            f"Splitting dataset: {100 - config.test_size:.1f}% train, {config.test_size:.1f}% test"
        )
        split_dataset = train_dataset.train_test_split(
            test_size=config.test_size, seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        eval_dataset = None

    # Student processor (prefix can be string, dict, or callable)
    student_processor = SpeechDistillDatasetProcessor(
        tokenizer=tokenizer,
        prefix=student_prefix,
        text_bos=config.text_bos,
        text_eos=config.text_eos,
        text_prefix=text_prefix,
        speech_bos=config.speech_bos,
        speech_eos=config.speech_eos,
        device=device,
        max_length=config.max_length,
    )

    # Teacher processor (prefix can be string, dict, or callable)
    teacher_processor = SpeechDistillDatasetProcessor(
        tokenizer=tokenizer,
        prefix=teacher_prefix,
        text_bos=config.text_bos,
        text_eos=config.text_eos,
        text_prefix=text_prefix,
        speech_bos=config.speech_bos,
        speech_eos=config.speech_eos,
        device=device,
        max_length=config.max_length,
    )

    # Create the picklable processor
    distill_processor = DistillationDataProcessor(student_processor, teacher_processor)

    # Use ProcessedDataCollator for already-processed data
    data_collator = ProcessedDataCollator(
        tokenizer,
        speech_bos=config.speech_bos,
        pad_token_id=153478,  # <|semantic_token_end|>
    )
    print(
        f"Using ProcessedDataCollator with on-the-fly processing (set_transform), speech_bos={config.speech_bos}"
    )

    # Apply on-the-fly transformation
    train_dataset.set_transform(distill_processor)
    if eval_dataset is not None:
        eval_dataset.set_transform(distill_processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=config.report_to,
        save_total_limit=3,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=(
            config.dataloader_prefetch_factor
            if config.dataloader_num_workers > 0
            else None
        ),
    )

    # Initialize Trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        temperature=config.temperature,
        alpha=config.alpha,
    )

    # Print a sample to verify processing
    print("\n" + "=" * 50)
    print("SAMPLE DATA PREVIEW")
    print("=" * 50)
    sample = train_dataset[0]

    if "student_input_ids" in sample:
        s_ids = sample["student_input_ids"]
        t_ids = sample["teacher_input_ids"]

        # Convert to list if they are tensors
        if torch.is_tensor(s_ids):
            s_ids = s_ids.tolist()
        if torch.is_tensor(t_ids):
            t_ids = t_ids.tolist()

        print(f"\n--- STUDENT INPUT (First 100 and Last 10 tokens) ---")
        print(
            f"Text: {tokenizer.decode(s_ids[:100])} ... {tokenizer.decode(s_ids[-10:])}"
        )
        print(f"IDs: {s_ids[:20]} ... {s_ids[-10:]}")

        print(f"\n--- TEACHER INPUT (First 100 and Last 10 tokens) ---")
        print(
            f"Text: {tokenizer.decode(t_ids[:100])} ... {tokenizer.decode(t_ids[-10:])}"
        )
        print(f"IDs: {t_ids[:20]} ... {t_ids[-10:]}")
    else:
        # Legacy format
        ids = sample["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        print(f"\n--- INPUT (First 100 and Last 10 tokens) ---")
        print(f"Text: {tokenizer.decode(ids[:100])} ... {tokenizer.decode(ids[-10:])}")
        print(f"IDs: {ids[:20]} ... {ids[-10:]}")
    print("=" * 50 + "\n")

    trainer.train()


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility with multiprocessing
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Distill a teacher LLM into a student LLM."
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Soul-AILab/SoulX-Podcast-1.7B-dialect",
        help="Teacher model ID or path",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="./pretrained_models/Qwen3-0.6B",
        help="Student model ID or path",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled_model",
        help="Output directory for the distilled model",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--teacher_prefix",
        type=str,
        default="<|task_podcast|><|SPEAKER_0|>",
        help="Prefix for teacher input (string or JSON dict for per-lang prefixes)",
    )
    parser.add_argument(
        "--student_prefix",
        type=str,
        default="",
        help="Prefix for student input (string or JSON dict for per-lang prefixes)",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use PEFT LoRA for student training",
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument(
        "--use_rslora",
        action="store_true",
        help="Whether to use Rank-Stabilized LoRA",
    )
    parser.set_defaults(use_rslora=True)
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="pissa",
        help="LoRA weight initialization method (e.g., 'pissa', 'gaussian', 'default')",
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for task loss vs distillation loss",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (default: True)",
    )
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--test_size",
        type=int,
        default=10,
        help="Number of samples for test split (default: 10)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Reporting platform for Trainer (e.g., 'wandb', 'tensorboard', 'none')",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of workers for data loading (on-the-fly processing)",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--text_bos",
        type=str,
        default="<|text_start|>",
        help="Text begin-of-sequence token",
    )
    parser.add_argument(
        "--text_eos",
        type=str,
        default="<|text_end|>",
        help="Text end-of-sequence token",
    )
    parser.add_argument(
        "--text_prefix",
        type=str,
        default='{"en": "", "zh": "", "yue": "<|Yue|>"}',
        help="Text prefix after text_eos, before speech tokens",
    )
    parser.add_argument(
        "--speech_bos",
        type=str,
        default="<|semantic_token_start|>",
        help="Speech begin-of-sequence token",
    )
    parser.add_argument(
        "--speech_eos",
        type=str,
        default="<|semantic_token_end|>",
        help="Speech end-of-sequence token",
    )

    main_args = parser.parse_args()
    train(main_args)
