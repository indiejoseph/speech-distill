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
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_model_id, trust_remote_code=True
    )

    # Get vocab sets
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()

    # Find missing tokens
    missing_tokens = [token for token in teacher_vocab if token not in student_vocab]
    print(f"Found {len(missing_tokens)} missing tokens in student vocab.")

    if len(missing_tokens) > 0:
        print("Adding missing tokens to student tokenizer...")
        student_tokenizer.add_tokens(missing_tokens)

    # Save the expanded tokenizer
    student_tokenizer.save_pretrained(output_dir)
    print(f"Expanded tokenizer saved to {output_dir}")

    # Load student model and resize embeddings
    print(f"Loading student model: {student_model_id}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    old_vocab_size = student_model.config.vocab_size
    print(
        f"Resizing student model embeddings from {old_vocab_size} to {len(student_tokenizer)}"
    )
    student_model.resize_token_embeddings(len(student_tokenizer))

    # Apply noisy mean initialization to new embeddings
    num_new_tokens = len(student_tokenizer) - old_vocab_size

    with torch.no_grad():
        if num_new_tokens > 0:
            print(
                f"Initializing {num_new_tokens} new token embeddings with noisy mean..."
            )
            input_embeddings = student_model.get_input_embeddings().weight
            _noisy_mean_initialization(input_embeddings, num_new_tokens)

            output_embeddings = student_model.get_output_embeddings().weight
            _noisy_mean_initialization(output_embeddings, num_new_tokens)
            print("New embeddings initialized successfully!")

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
