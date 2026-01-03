import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _noisy_mean_initialization(
    embed_weight: "torch.Tensor", num_new_tokens: int
) -> None:
    """Initialize new token embeddings with mean + Gaussian noise.

    This is the default initialization method used by LlamaFactory.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix
    """
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def expand_student_vocab(teacher_model_id, student_model_id, output_dir):
    print(f"Loading teacher tokenizer: {teacher_model_id}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_id, trust_remote_code=True
    )

    print(f"Loading student tokenizer: {student_model_id}")
    student_tokenizer_old = AutoTokenizer.from_pretrained(
        student_model_id, trust_remote_code=True
    )

    # We will use the teacher's tokenizer as the new student tokenizer
    # to ensure perfect ID alignment.
    print("Using teacher tokenizer as the new student tokenizer...")
    new_student_tokenizer = teacher_tokenizer
    new_student_tokenizer.save_pretrained(output_dir)

    # Load student model
    print(f"Loading student model: {student_model_id}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    old_vocab_size = student_model.config.vocab_size
    new_vocab_size = len(new_student_tokenizer)

    print(
        f"Resizing student model embeddings from {old_vocab_size} to {new_vocab_size}"
    )

    # Get old embeddings
    old_input_embeds = student_model.get_input_embeddings().weight.detach()
    old_output_embeds = student_model.get_output_embeddings().weight.detach()
    embedding_dim = old_input_embeds.size(1)

    # Create new embedding matrices
    new_input_embeds = torch.zeros(
        (new_vocab_size, embedding_dim), dtype=old_input_embeds.dtype
    )
    new_output_embeds = torch.zeros(
        (new_vocab_size, embedding_dim), dtype=old_output_embeds.dtype
    )

    # Initialize with noisy mean (for tokens we don't find in the old vocab)
    avg_input = old_input_embeds.mean(dim=0, keepdim=True)
    avg_output = old_output_embeds.mean(dim=0, keepdim=True)

    # Fill with noisy mean first
    new_input_embeds.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    new_input_embeds += avg_input
    new_output_embeds.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    new_output_embeds += avg_output

    # Map old embeddings to new indices
    print("Mapping old embeddings to new indices...")
    old_vocab = student_tokenizer_old.get_vocab()
    new_vocab = new_student_tokenizer.get_vocab()

    matched_count = 0
    for token, new_idx in new_vocab.items():
        if token in old_vocab:
            old_idx = old_vocab[token]
            if old_idx < old_vocab_size:
                new_input_embeds[new_idx] = old_input_embeds[old_idx]
                new_output_embeds[new_idx] = old_output_embeds[old_idx]
                matched_count += 1

    print(
        f"Matched and preserved {matched_count} tokens from the original student model."
    )

    # Update model with new embeddings
    student_model.resize_token_embeddings(new_vocab_size)
    student_model.get_input_embeddings().weight.data = new_input_embeds
    student_model.get_output_embeddings().weight.data = new_output_embeds
    student_model.config.vocab_size = new_vocab_size

    # Save the resized model
    student_model.save_pretrained(output_dir)
    print(f"Resized student model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand student model vocabulary to match teacher model"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="Soul-AILab/SoulX-Podcast-1.7B-dialect",
        help="Teacher model ID from Hugging Face Hub",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Student model ID from Hugging Face Hub",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pretrained_models/Qwen3-0.6B",
        help="Directory to save the expanded student model and tokenizer",
    )

    args = parser.parse_args()

    expand_student_vocab(args.teacher_model, args.student_model, args.output_dir)
