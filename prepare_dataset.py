"""
Prepare dataset by applying DistillationDataProcessor.
This pre-processes the dataset once, saving time during training.
"""

import argparse
import os
import json
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from data import (
    SpeechDistillDatasetProcessor,
    DistillationDataProcessor,
    align_prefixes,
    parse_prefix,
)


def prepare_dataset(config):
    device = "cuda" if config.device == "cuda" else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.student_model, trust_remote_code=True
    )

    # Set pad_token
    if config.pad_token:
        if config.pad_token not in tokenizer.get_vocab():
            raise ValueError(
                f"Specified pad_token '{config.pad_token}' not found in tokenizer vocabulary. "
                f"Please ensure the token exists or use a different one."
            )
        tokenizer.pad_token = config.pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parse prefixes
    teacher_prefix = parse_prefix(config.teacher_prefix)
    student_prefix = parse_prefix(config.student_prefix)
    text_prefix = parse_prefix(config.text_prefix)

    # Align prefixes
    print("Aligning teacher and student prefixes...")
    teacher_prefix, student_prefix = align_prefixes(
        teacher_prefix, student_prefix, tokenizer
    )

    # Load dataset
    print(f"Loading dataset from: {config.dataset_path}")
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
        dataset = raw_dataset.get("train", raw_dataset)
    else:
        dataset = raw_dataset

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")

    # Create processors
    print("Creating DistillationDataProcessor for on-the-fly conversion...")
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

    distill_processor = DistillationDataProcessor(student_processor, teacher_processor)

    # Apply transformation
    print("Processing dataset (this may take a while)...")
    processed_dataset = dataset.map(
        distill_processor,
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_proc,
        desc="Processing dataset",
    )

    # Save output
    output_path = config.output_path
    print(f"Saving processed dataset to: {output_path}")
    processed_dataset.save_to_disk(output_path)

    print(f"\n✓ Dataset preprocessing complete!")
    print(f"  - Processed: {len(processed_dataset)} examples")
    print(f"  - Columns: {processed_dataset.column_names}")
    print(f"  - Location: {output_path}")
    print(f"\nYou can now use this dataset in train.py with:")
    print(f"  python train.py --dataset_path {output_path} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset with DistillationDataProcessor"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the raw dataset (HF ID or local path)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the processed dataset",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="./pretrained_models/Qwen3-0.6B",
        help="Student model ID or path (used for tokenizer)",
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
    parser.add_argument(
        "--pad_token",
        type=str,
        default="<|semantic_token_end|>",
        help="Padding token for the model",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing during dataset mapping (higher = faster but more memory)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel processing (set >1 carefully with GPU)",
    )

    args = parser.parse_args()
    prepare_dataset(args)
