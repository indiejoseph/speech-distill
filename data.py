import json
import torch
import numpy as np
from typing import Union, Dict, List, Any, Optional, Callable, TYPE_CHECKING
from utils import prepare_inputs, prepare_inputs_batch

if TYPE_CHECKING:
    from transformers import AutoTokenizer


def parse_prefix(prefix_str):
    if not prefix_str:
        return ""
    try:
        return json.loads(prefix_str)
    except json.JSONDecodeError:
        return prefix_str


def align_prefixes(teacher_prefix, student_prefix, tokenizer):
    """
    Align teacher and student prefixes by left-padding the shorter one with pad_token.
    Ensures both prefixes have the same number of tokens.
    """
    pad_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token

    def _align_single(t_p, s_p):
        t_ids = tokenizer.encode(t_p, add_special_tokens=False)
        s_ids = tokenizer.encode(s_p, add_special_tokens=False)

        if len(t_ids) == len(s_ids):
            return t_p, s_p

        max_len = max(len(t_ids), len(s_ids))

        if len(t_ids) < max_len:
            t_p = (pad_token * (max_len - len(t_ids))) + t_p
        if len(s_ids) < max_len:
            s_p = (pad_token * (max_len - len(s_ids))) + s_p

        return t_p, s_p

    if isinstance(teacher_prefix, dict) or isinstance(student_prefix, dict):
        # Handle dict case (per-language prefixes)
        if isinstance(teacher_prefix, str):
            teacher_prefix = {"default": teacher_prefix}
        if isinstance(student_prefix, str):
            student_prefix = {"default": student_prefix}

        all_keys = set(teacher_prefix.keys()) | set(student_prefix.keys())
        new_t = {}
        new_s = {}
        for k in all_keys:
            t_val = teacher_prefix.get(k, teacher_prefix.get("default", ""))
            s_val = student_prefix.get(k, student_prefix.get("default", ""))
            new_t[k], new_s[k] = _align_single(t_val, s_val)
        return new_t, new_s
    else:
        # Handle string case
        return _align_single(teacher_prefix, student_prefix)


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
        pad_token_id=153478,  # <|semantic_token_end|>
        speech_bos: str = "<|semantic_token_start|>",
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.speech_bos = speech_bos

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
        batch = self._pad_sequences(student_input_ids, student_attention_mask)

        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.pad_token_id] = -100

        # Add teacher inputs if they exist
        if teacher_input_ids and teacher_input_ids[0] is not None:
            teacher_batch = self._pad_sequences(
                teacher_input_ids, teacher_attention_mask
            )
            batch["teacher_input_ids"] = teacher_batch["input_ids"]
            batch["teacher_attention_mask"] = teacher_batch["attention_mask"]

        # Mask out text tokens (tokens before speech_bos)
        speech_mask = self._create_speech_token_mask(batch["input_ids"])
        if speech_mask is not None:
            batch["labels"][speech_mask == 0] = -100

        return batch

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
                    torch.full(
                        (padding_length,), self.pad_token_id, dtype=input_ids.dtype
                    ),
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
        Create a mask marking positions from self.speech_bos onwards.
        This is used to focus KL divergence loss on speech token prediction only.

        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs

        Returns:
            [batch_size, seq_len] binary mask where 1 = speech token position, 0 = text
        """
        try:
            # Get speech_bos token ID
            token_ids = self.tokenizer.encode(self.speech_bos, add_special_tokens=False)
            if not token_ids:
                return None
            speech_bos_token_id = token_ids[0]

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


class DistillationDataProcessor:
    """
    Picklable wrapper for dataset transformation.
    """

    def __init__(self, student_processor, teacher_processor):
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor

    def __call__(self, examples):
        """Process examples for both student and teacher."""
        # Check if we are receiving a batch or a single example
        # In HF datasets, if batched=True, examples is a dict of lists
        is_batched = isinstance(examples.get("text", examples.get("audio")), list)

        if is_batched:
            # Process for student
            student_inputs = self.student_processor.process_batch(examples)
            # Process for teacher
            teacher_inputs = self.teacher_processor.process_batch(examples)

            return {
                "student_input_ids": student_inputs["input_ids"],
                "student_attention_mask": student_inputs["attention_mask"],
                "teacher_input_ids": teacher_inputs["input_ids"],
                "teacher_attention_mask": teacher_inputs["attention_mask"],
            }
        else:
            # Single example logic
            example = examples
            # Get audio - can be path, dict with 'array'/'sampling_rate', or numpy array
            audio_input = example.get("audio")

            # Convert numpy array to tensor if needed (but keep dict format for resampling info)
            if isinstance(audio_input, dict):
                # HuggingFace format with 'array' and 'sampling_rate' - keep as dict for resampling
                if isinstance(audio_input.get("array"), np.ndarray):
                    audio_array = torch.from_numpy(audio_input["array"]).float()
                    audio_input = {
                        "array": audio_array,
                        "sampling_rate": audio_input.get("sampling_rate", 16000),
                    }
            elif isinstance(audio_input, np.ndarray):
                # Convert numpy array to tensor
                audio_input = torch.from_numpy(audio_input).float()

            text = example.get("text", "")
            lang = example.get("lang", "")

            # Process for student
            student_inputs = self.student_processor.process_example(
                {
                    "audio": audio_input,
                    "text": text,
                    "lang": lang,
                }
            )

            # Process for teacher
            teacher_inputs = self.teacher_processor.process_example(
                {
                    "audio": audio_input,
                    "text": text,
                    "lang": lang,
                }
            )

            return {
                "student_input_ids": student_inputs["input_ids"],
                "student_attention_mask": student_inputs["attention_mask"],
                "teacher_input_ids": teacher_inputs["input_ids"],
                "teacher_attention_mask": teacher_inputs["attention_mask"],
            }
