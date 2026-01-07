"""
Stage 1: Text-to-Speech Token Alignment Training

This script fine-tunes a model to align text with new speech tokens using SFTTrainer from TRL.
All model weights are frozen except for the new speech tokens, allowing the model to learn
the mapping between text and speech tokens without catastrophic forgetting.

Usage:
    python stage1.py \
        --model_path <path_to_model> \
        --dataset_path <path_to_dataset> \
        --output_dir <output_directory> \
        --num_new_tokens <number_of_new_speech_tokens>
"""

import torch
import argparse
import os

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer, SFTConfig
from data import parse_prefix, SpeechDistillDatasetProcessor


def freeze_model_weights(model, num_new_tokens):
    """
    Freeze all model weights except the embeddings of newly added speech tokens.
    Uses gradient masking to prevent updates to old token embeddings.

    Args:
        model: The language model to freeze
        num_new_tokens: Number of new tokens added to the model's vocabulary
    """
    # Freeze all parameters by default
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze only the new token embeddings in input and output layers
    if num_new_tokens > 0:
        # Get the old vocab size (total vocab - new tokens)
        input_embeddings = model.get_input_embeddings()
        old_vocab_size = input_embeddings.weight.size(0) - num_new_tokens

        # Unfreeze input embeddings (entire layer trainable, gradients masked via hook)
        if input_embeddings is not None:
            input_embeddings.weight.requires_grad_(True)

            # Register backward hook to zero out gradients for old tokens
            def mask_input_grads(grad):
                if grad is not None:
                    grad = grad.clone()
                    grad[:old_vocab_size] = 0.0  # Zero gradients for old tokens
                return grad

            input_embeddings.weight.register_hook(mask_input_grads)

        # Unfreeze output embeddings (entire layer trainable, gradients masked via hook)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight.requires_grad_(True)

            # Register backward hook to zero out gradients for old tokens
            def mask_output_grads(grad):
                if grad is not None:
                    grad = grad.clone()
                    grad[:old_vocab_size] = 0.0  # Zero gradients for old tokens
                return grad

            output_embeddings.weight.register_hook(mask_output_grads)

    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    new_token_params = num_new_tokens * model.get_input_embeddings().weight.size(1)

    print(f"\n{'='*60}")
    print(f"Trainable Parameters Summary (Gradient Masking Applied):")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (by requires_grad): {trainable_params:,}")
    print(f"Effectively trainable (new tokens only): {new_token_params:,}")
    print(f"Frozen parameters: {total_params - new_token_params:,}")
    print(f"Trainable ratio: {100 * new_token_params / total_params:.4f}%")
    print(f"Note: Gradient masking applied to embedding and lm_head layers.")
    print(
        f"Old tokens ({model.get_input_embeddings().weight.size(0) - num_new_tokens:,}) "
        f"will not receive gradient updates."
    )
    print(f"{'='*60}\n")


def train_stage1(config):
    """
    Train the model for text-to-speech token alignment using SFTTrainer.

    Args:
        config: Training configuration object from argparse
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {config.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Load tokenizer
    print(f"Loading tokenizer from: {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )

    # Remove chat template
    tokenizer.chat_template = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine number of new tokens
    # New speech tokens are typically in the range specified by token range
    # For example, tokens like <|semantic_token_start|> = 153477, <|semantic_token_end|> = 153478
    # and speech tokens in between
    if hasattr(config, "num_new_tokens") and config.num_new_tokens > 0:
        num_new_tokens = config.num_new_tokens
    else:
        # Default: assume new tokens were added at the end
        num_new_tokens = 1000  # Adjust based on your actual setup

    # Freeze all weights except new speech tokens
    print(
        f"\nFreezing model weights, keeping {num_new_tokens} new speech tokens unfrozen..."
    )
    freeze_model_weights(model, num_new_tokens)

    # Parse and align prefix
    print("\nProcessing prefix configuration...")
    prefix = parse_prefix(config.prefix)
    text_prefix = parse_prefix(config.text_prefix)

    # If prefix is a dict (per-language), we'll use it as-is
    if isinstance(prefix, dict):
        print(f"Using per-language prefixes: {list(prefix.keys())}")
    else:
        print(f"Using prefix: {prefix if prefix else '(empty)'}")

    # Load dataset
    print(f"\nLoading dataset from: {config.dataset_path}")
    if os.path.exists(config.dataset_path):
        dataset = load_from_disk(config.dataset_path)
    else:
        dataset = load_dataset(config.dataset_path)

    # Handle train/test split
    if isinstance(dataset, dict):
        train_dataset = dataset.get("train", dataset)
    else:
        train_dataset = dataset

    print(f"Dataset loaded: {len(train_dataset)} examples")

    # Split into train and eval if needed
    if config.eval_size > 0:
        print(
            f"Splitting dataset: {100 - config.eval_size:.1f}% train, {config.eval_size:.1f}% eval"
        )
        split_dataset = train_dataset.train_test_split(
            test_size=config.eval_size, seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        eval_dataset = None

    # Create processor for dataset (to extract speech tokens from audio and convert to input_ids)
    print("\nCreating dataset processor for audio-to-tokens conversion...")
    processor = SpeechDistillDatasetProcessor(
        tokenizer=tokenizer,
        prefix=prefix,
        text_bos=config.text_bos,
        text_eos=config.text_eos,
        text_prefix=text_prefix,
        speech_bos=config.speech_bos,
        speech_eos=config.speech_eos,
        device=device,
    )

    # Process datasets to create text field for SFTTrainer
    print("\nProcessing dataset for SFTTrainer...")
    print(
        "Using SpeechDistillDatasetProcessor to convert audio to tokens, then to text"
    )

    def format_for_sft(batch):
        """Format batch of examples as text for SFTTrainer to tokenize.

        Uses SpeechDistillDatasetProcessor to convert audio to input_ids (including speech tokens),
        then decodes back to text for SFTTrainer.

        Args:
            batch: A batch dictionary with lists of examples (batched=True)
        """
        texts = []
        for i in range(len(batch.get(list(batch.keys())[0], []))):
            try:
                # Extract single example from batch
                example = {key: batch[key][i] for key in batch.keys()}

                # Use processor to convert audio + text to input_ids
                # This includes speech tokens extracted from audio
                processed = processor.process_example(example)

                # Get input_ids from processor output
                input_ids = processed.get("input_ids", [])
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.tolist()

                # Convert input_ids back to text using tokenizer
                # This preserves both text and speech tokens in text form
                text_content = tokenizer.decode(input_ids)

                if not text_content or not text_content.strip():
                    texts.append("")
                else:
                    texts.append(text_content)
            except Exception as e:
                # Fallback: try with just text and no audio
                try:
                    text = example.get("text", "").strip()
                    if text:
                        texts.append(text)
                    else:
                        texts.append("")
                except:
                    texts.append("")

        return {"text": texts}

    # Apply formatting to datasets
    print("Formatting train dataset...")
    train_dataset = train_dataset.map(
        format_for_sft,
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset",
    )

    if eval_dataset is not None:
        print("Formatting eval dataset...")
        eval_dataset = eval_dataset.map(
            format_for_sft,
            batched=True,
            batch_size=32,
            remove_columns=eval_dataset.column_names,
            desc="Formatting eval dataset",
        )

    # Filter out empty examples
    print("Filtering empty examples...")
    train_dataset = train_dataset.filter(
        lambda x: len(x.get("text", "")) > 0,
        desc="Filtering train dataset",
    )

    if eval_dataset is not None:
        eval_dataset = eval_dataset.filter(
            lambda x: len(x.get("text", "")) > 0,
            desc="Filtering eval dataset",
        )

    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval dataset size: {len(eval_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after processing!")

    # Setup training arguments
    training_args = SFTConfig(
        output_dir=config.output_dir,
        max_length=4096,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=eval_dataset is not None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        use_liger_kernel=True,  # Disabled because liger-kernel not installed
        optim="adamw_8bit" if config.use_8bit_optimizer else "adamw_torch",
        weight_decay=config.weight_decay,
        seed=config.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=config.num_workers,
        report_to=["wandb"] if config.use_wandb else [],
        remove_unused_columns=False,
        packing=True,  # Disable packing for now to avoid attention implementation issues
        dataset_text_field="text",  # Tell SFTTrainer which field contains text to tokenize
    )

    # Initialize trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # SFTTrainer uses processing_class for tokenization
    )

    # Train
    print("\nStarting training...")
    print(f"Output directory: {config.output_dir}")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))

    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Text-to-Speech Token Alignment Training"
    )

    # Model and data paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to fine-tune",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset (local or HuggingFace)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and final model",
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging frequency",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint frequency",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluation frequency",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.1,
        help="Evaluation set size (0-1)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Model freezing
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=8220,
        help="Number of new speech tokens added to the model",
    )

    # Text and speech token configuration
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add before text (string or JSON dict for per-language prefixes)",
    )
    parser.add_argument(
        "--text_bos",
        type=str,
        default="<|text_start|>",
        help="Text beginning-of-sequence token",
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
        help="Text prefix after eos, before speech tokens (string or JSON dict for per-language prefixes)",
    )
    parser.add_argument(
        "--speech_bos",
        type=str,
        default="<|semantic_token_start|>",
        help="Speech beginning-of-sequence token",
    )
    parser.add_argument(
        "--speech_eos",
        type=str,
        default="<|semantic_token_end|>",
        help="Speech end-of-sequence token",
    )

    # Optimization options
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="Use 8-bit AdamW optimizer",
    )
    parser.set_defaults(use_8bit_optimizer=False)
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train_stage1(args)


if __name__ == "__main__":
    main()
