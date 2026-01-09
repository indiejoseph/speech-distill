import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import (
    ProcessedDataCollator,
    SpeechDistillDatasetProcessor,
    parse_prefix,
)


def extract_teacher_logprobs(config):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Teacher
    print(f"Loading teacher model from: {config.teacher_model_path}")

    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # Get the device of the model
    # When using device_map="auto", model.device might be the first device
    device = getattr(teacher, "device", next(teacher.parameters()).device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher.eval()

    # 2. Load Dataset
    print(f"Loading dataset from: {config.dataset_path}")
    if os.path.exists(config.dataset_path):
        dataset = load_from_disk(config.dataset_path)
    else:
        dataset = load_dataset(config.dataset_path, split=config.dataset_split)

    # 3. Setup Processor
    teacher_prefix = parse_prefix(config.teacher_prefix)
    text_prefix = parse_prefix(config.text_prefix)

    processor = SpeechDistillDatasetProcessor(
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

    # Apply transform if dataset doesn't already have input_ids
    if "input_ids" not in dataset.column_names:

        def transform_fn(examples):
            # Check if examples is batched (dict of lists) or single (dict)
            is_batched = isinstance(examples.get("text", examples.get("audio")), list)
            if is_batched:
                return processor.process_batch(examples)
            else:
                return processor.process_example(examples)

        dataset.set_transform(transform_fn)

    # 4. Setup Collator and DataLoader
    collator = ProcessedDataCollator(tokenizer=tokenizer, speech_bos=config.speech_bos)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,  # CRITICAL: Must be False to preserve dataset order
    )

    # 5. Extraction Loop
    top_k = config.top_k
    all_top_v = []
    all_top_i = []

    print(f"Starting extraction (Top-{top_k})...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Use teacher_input_ids if present (from collator), else fallback to input_ids
            input_ids = batch.get("teacher_input_ids", batch["input_ids"]).to(device)
            attention_mask = batch.get(
                "teacher_attention_mask", batch["attention_mask"]
            ).to(device)

            # Forward pass through teacher
            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]

            # Convert to logprobs (logspace is better for KL)
            logprobs = F.log_softmax(logits, dim=-1)

            # Extract Top-K values and indices
            top_v, top_i = torch.topk(logprobs, k=top_k, dim=-1)

            # Truncate to actual sequence lengths (so they match the unpadded dataset items)
            lengths = attention_mask.sum(dim=1).cpu().tolist()
            for b_idx in range(len(lengths)):
                actual_len = int(lengths[b_idx])
                # Save as fp16/int32 to reduce disk usage
                all_top_v.append(
                    top_v[b_idx, :actual_len].to(torch.float16).cpu().numpy()
                )
                all_top_i.append(
                    top_i[b_idx, :actual_len].to(torch.int32).cpu().numpy()
                )

    # 5. Enrich Dataset
    print("Enriching dataset with teacher logprobs...")
    if len(all_top_v) != len(dataset):
        print(
            f"Error: Alignment mismatch! Extracted {len(all_top_v)} but dataset has {len(dataset)}"
        )
        return

    # Add columns to the original dataset
    dataset = dataset.add_column("teacher_top_k_v", all_top_v)
    dataset = dataset.add_column("teacher_top_k_i", all_top_i)

    # 6. Save
    print(f"Saving enriched dataset to: {config.output_path}")
    dataset.save_to_disk(config.output_path)
    print("Successfully completed extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract teacher logprobs for distillation"
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        required=True,
        help="Path to teacher model checkpoint",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to tokenized dataset"
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save enriched dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Inference batch size"
    )
    parser.add_argument(
        "--top_k", type=int, default=100, help="Number of logits to keep per token"
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="Max sequence length"
    )
    parser.add_argument(
        "--teacher_prefix",
        type=str,
        default="<|task_podcast|><|SPEAKER_0|>",
        help="Prefix for teacher input",
    )
    parser.add_argument(
        "--text_prefix",
        type=str,
        default='{"en": "", "zh": "", "yue": "<|Yue|>"}',
        help="Text prefix",
    )
    parser.add_argument(
        "--text_bos",
        type=str,
        default="<|text_start|>",
        help="Text bos",
    )
    parser.add_argument(
        "--text_eos",
        type=str,
        default="<|text_end|>",
        help="Text eos",
    )
    parser.add_argument(
        "--speech_bos",
        type=str,
        default="<|semantic_token_start|>",
        help="Speech bos",
    )
    parser.add_argument(
        "--speech_eos",
        type=str,
        default="<|semantic_token_end|>",
        help="Speech eos",
    )

    args = parser.parse_args()
    extract_teacher_logprobs(args)
