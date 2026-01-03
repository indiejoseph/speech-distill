import os
import torch
from typing import Union, Dict, List, Any, Optional, Callable, TYPE_CHECKING
from datasets import load_dataset, load_from_disk
from utils import prepare_inputs, prepare_inputs_batch

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset


class SpeechDistillDataset:
    dataset: Union[
        "DatasetDict",
        "Dataset",
        "IterableDatasetDict",
        "IterableDataset",
        Dict[str, Any],
    ]

    def __init__(
        self,
        dataset_name_or_path: str,
        tokenizer: "AutoTokenizer",
        max_length: int = 512,
        teacher_prefix: Union[str, Dict[str, str]] = "",
        student_prefix: Union[str, Dict[str, str]] = "",
        test_size: float = 0.1,
        seed: int = 42,
    ):
        if os.path.exists(dataset_name_or_path):
            self.dataset = load_from_disk(dataset_name_or_path)
        else:
            self.dataset = load_dataset(dataset_name_or_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_prefix = teacher_prefix
        self.student_prefix = student_prefix

        # Check if dataset has splits
        if hasattr(self.dataset, "keys"):
            # Dataset already has splits
            self.has_splits = True
        else:
            # No splits, create train/test split
            self.has_splits = False
            self.test_size = test_size
            self.seed = seed
            self._create_splits()

    def _create_splits(self):
        """Create train/test splits if they don't exist."""
        from datasets import Dataset

        if isinstance(self.dataset, Dataset):
            splits = self.dataset.train_test_split(
                test_size=self.test_size,
                seed=self.seed,
            )

            self.dataset = {
                "train": splits["train"],
                "test": splits["test"],
            }
            self.has_splits = True
        else:
            raise ValueError(
                f"Cannot create splits for dataset type {type(self.dataset)}. "
                "train_test_split is only available for Dataset objects."
            )

    def get_split(self, split: str = "train") -> Any:
        if self.has_splits:
            return self.dataset[split]
        else:
            # Should not reach here after __init__, but handle anyway
            return self.dataset


class SpeechDistillDataCollator:
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        max_length: int = 512,
        teacher_prefix: Union[str, Dict[str, str]] = "",
        student_prefix: Union[str, Dict[str, str]] = "",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_prefix = teacher_prefix
        self.student_prefix = student_prefix

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # features is a list of dicts with ['audio', 'lang', 'text', 'speech_tokens']

        teacher_texts = []
        student_texts = []
        for f in features:
            lang = f["lang"]
            lang_tag = f"[{lang.upper()}]"
            speech_tokens_str = " ".join([str(t) for t in f["speech_tokens"]])
            content = f"{lang_tag} {speech_tokens_str} {f['text']}"

            # Handle teacher prefix (str or dict)
            t_prefix = self.teacher_prefix
            if isinstance(t_prefix, dict):
                t_prefix = t_prefix.get(lang, t_prefix.get("default", ""))

            # Handle student prefix (str or dict)
            s_prefix = self.student_prefix
            if isinstance(s_prefix, dict):
                s_prefix = s_prefix.get(lang, s_prefix.get("default", ""))

            teacher_texts.append(t_prefix + content)
            student_texts.append(s_prefix + content)

        # Tokenize student inputs
        student_batch = self.tokenizer(
            student_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Tokenize teacher inputs
        teacher_batch = self.tokenizer(
            teacher_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # For CausalLM, labels are usually the same as input_ids
        student_batch["labels"] = student_batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            student_batch["labels"][
                student_batch["labels"] == self.tokenizer.pad_token_id
            ] = -100

        # Add teacher inputs to the batch
        student_batch["teacher_input_ids"] = teacher_batch["input_ids"]
        student_batch["teacher_attention_mask"] = teacher_batch["attention_mask"]

        return student_batch


class SpeechDistillDatasetProcessor:
    """
    Dataset processor that:
    1. Converts audio to speech tokens using get_speech_tokens()
    2. Converts text + speech tokens to model inputs using prepare_inputs()

    Args:
        prefix: Can be a string, dict mapping lang->prefix, or callable(text, lang) -> str
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        prefix: Union[str, Dict[str, str], Callable[[str, str], str]] = "",
        text_bos: str = "<|text_start|>",
        text_eos: str = "<|text_end|>",
        text_prefix: Union[str, Dict[str, str], Callable[[str, str], str]] = "",
        speech_bos: str = "<|semantic_token_start|>",
        speech_eos: str = "<|semantic_token_end|>",
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.text_bos = text_bos
        self.text_eos = text_eos
        self.text_prefix = text_prefix
        self.speech_bos = speech_bos
        self.speech_eos = speech_eos
        self.device = device

    def _get_prefix(self, text: str, lang: str = "") -> str:
        """Get prefix based on text and language."""
        if callable(self.prefix):
            return self.prefix(text, lang)
        elif isinstance(self.prefix, dict):
            return self.prefix.get(lang, self.prefix.get("default", ""))
        else:
            return self.prefix

    def _get_text_prefix(self, text: str, lang: str = "") -> str:
        """Get text_prefix based on text and language."""
        if callable(self.text_prefix):
            return self.text_prefix(text, lang)
        elif isinstance(self.text_prefix, dict):
            return self.text_prefix.get(lang, self.text_prefix.get("default", ""))
        else:
            return self.text_prefix

    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single dataset example.

        Args:
            example: Dict containing 'audio' (file path, numpy array, or HF dict with 'array'/'sampling_rate'),
                    'text', and optionally 'lang' keys

        Returns:
            Dict with tokenized inputs ready for model
        """
        audio_input = example.get("audio")
        # audio_input can be:
        # - str: file path
        # - dict: HuggingFace format with 'array' and 'sampling_rate'
        # - numpy.ndarray: raw audio samples
        if audio_input is None:
            raise ValueError("'audio' key not found in example")

        text = example.get("text", "")
        lang = example.get("lang", "")

        # Get prefix based on text and lang
        prefix = self._get_prefix(text, lang)
        text_prefix = self._get_text_prefix(text, lang)

        # Convert audio to speech tokens and prepare inputs
        model_inputs = prepare_inputs(
            text=text,
            audio_input=audio_input,
            prefix=prefix,
            text_bos=self.text_bos,
            text_eos=self.text_eos,
            text_prefix=text_prefix,
            speech_bos=self.speech_bos,
            speech_eos=self.speech_eos,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
        }

    def process_batch(
        self, examples: Dict[str, List[Any]]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Process a batch of dataset examples (for use with datasets.map()).

        Args:
            examples: Dict with lists of 'audio' (file path, numpy array, or HF dict with 'array'/'sampling_rate'),
                     'text', and optionally 'lang'

        Returns:
            Dict with lists of tokenized inputs
        """
        audio_inputs = examples.get("audio", examples.get("wav_path", []))
        texts = examples.get("text", [""] * len(audio_inputs))
        langs = examples.get("lang", [""] * len(audio_inputs))

        prefixes = [self._get_prefix(t, l) for t, l in zip(texts, langs)]
        text_prefixes = [self._get_text_prefix(t, l) for t, l in zip(texts, langs)]

        # Convert audio to speech tokens and prepare inputs in batch
        model_inputs = prepare_inputs_batch(
            texts=texts,
            audio_inputs=audio_inputs,
            prefixes=prefixes,
            text_bos=self.text_bos,
            text_eos=self.text_eos,
            text_prefixes=text_prefixes,
            speech_bos=self.speech_bos,
            speech_eos=self.speech_eos,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Return as lists of tensors (not padded yet)
        return {
            "input_ids": [ids for ids in model_inputs["input_ids"]],
            "attention_mask": [mask for mask in model_inputs["attention_mask"]],
        }


class ProcessedDataCollator:
    """
    Collator for data that has already been processed by SpeechDistillDatasetProcessor.
    Handles padding and creating labels for both student and teacher models.
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate student and teacher inputs if they exist
        student_input_ids = [
            f["student_input_ids"] for f in features if "student_input_ids" in f
        ]
        student_attention_mask = [
            f["student_attention_mask"]
            for f in features
            if "student_attention_mask" in f
        ]
        teacher_input_ids = [
            f.get("teacher_input_ids") for f in features if "teacher_input_ids" in f
        ]
        teacher_attention_mask = [
            f.get("teacher_attention_mask")
            for f in features
            if "teacher_attention_mask" in f
        ]

        # Handle case where we only have student inputs (no separate teacher)
        if not student_input_ids:
            student_input_ids = [f["input_ids"] for f in features]
            student_attention_mask = [f["attention_mask"] for f in features]

        # Pad student inputs
        student_batch = self._pad_sequences(student_input_ids, student_attention_mask)

        # Create labels (same as input_ids for causal LM)
        student_batch["labels"] = student_batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            student_batch["labels"][
                student_batch["labels"] == self.tokenizer.pad_token_id
            ] = -100

        # Add teacher inputs if they exist
        if teacher_input_ids and teacher_input_ids[0] is not None:
            teacher_batch = self._pad_sequences(
                teacher_input_ids, teacher_attention_mask
            )
            student_batch["teacher_input_ids"] = teacher_batch["input_ids"]
            student_batch["teacher_attention_mask"] = teacher_batch["attention_mask"]

        # Create speech token mask (mark positions from <|speech_bos|> onwards)
        # This helps focus KL divergence on speech token prediction only
        speech_token_mask = self._create_speech_token_mask(student_batch["input_ids"])
        if speech_token_mask is not None:
            student_batch["speech_token_mask"] = speech_token_mask

        return student_batch

    def _pad_sequences(self, input_ids_list, attention_mask_list):
        """Pad sequences to the same length."""
        max_length = max(len(ids) for ids in input_ids_list)

        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_input_ids = []
        batch_attention_mask = []

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            # Convert to tensor if they're lists
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            padding_length = max_length - len(input_ids)

            # Pad input_ids
            padded_input_ids = torch.cat(
                [
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
                ]
            )

            # Pad attention_mask
            padded_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype),
                ]
            )

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
        }

    def _create_speech_token_mask(
        self, input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Create a mask marking positions from <|semantic|> or <|speech_bos|> onwards.
        This is used to focus KL divergence loss on speech token prediction only.

        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs

        Returns:
            [batch_size, seq_len] binary mask where 1 = speech token position, 0 = text
        """
        try:
            # Try to find speech_bos token ID
            speech_bos_candidates = ["<|semantic|>", "<|speech_bos|>", "<|speech|>"]
            speech_bos_token_id = None

            for candidate in speech_bos_candidates:
                try:
                    token_ids = self.tokenizer.encode(
                        candidate, add_special_tokens=False
                    )
                    if token_ids:
                        speech_bos_token_id = token_ids[0]
                        break
                except:
                    continue

            if speech_bos_token_id is None:
                # If we can't find speech token, return None (use entire sequence)
                return None

            # Create mask: 1 where token appears onwards, 0 before
            batch_size, seq_len = input_ids.shape
            speech_mask = torch.zeros_like(input_ids, dtype=torch.float32)

            for i in range(batch_size):
                # Find first occurrence of speech_bos token
                positions = (input_ids[i] == speech_bos_token_id).nonzero(
                    as_tuple=True
                )[0]
                if len(positions) > 0:
                    first_speech_pos = positions[0].item()
                    # Mark all positions from first_speech_pos onwards as 1
                    speech_mask[i, first_speech_pos:] = 1.0

            return speech_mask
        except Exception as e:
            # If mask creation fails, return None and use entire sequence
            return None


def get_dataloader_info() -> None:
    print("Dataset columns: ['audio', 'lang', 'text', 'speech_tokens']")
    print("Supported languages: 'en', 'yue', 'zh'")
