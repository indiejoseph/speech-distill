from typing import TYPE_CHECKING, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from cosyvoice2.modeling import CosyVoice2


class CosyVoiceTeacherWrapper(nn.Module):
    """
    Wrapper for CosyVoice2 (Qwen2LM) to make it compatible with standard
    DistillationTrainer.

    This wrapper handles the complex input formatting (SOS, Task ID, Text, Speech splitting)
    and ensures that the forward pass correctly utilizes the teacher's specialized
    embedding and decoder layers.
    """

    def __init__(self, teacher_model: "CosyVoice2", tokenizer, text_vocab_size=152704):
        super().__init__()
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.text_vocab_size = text_vocab_size
        self.special_token_offset = text_vocab_size
        self.speech_token_offset = text_vocab_size + 2

        # Internal components mapping (based on COSYVOICE2.md)
        self.llm_embedding = teacher_model.llm_embedding
        self.base_llm = teacher_model.llm
        self.llm_decoder = teacher_model.llm_decoder
        self.speech_embedding = teacher_model.speech_embedding

        # Resolve the transformer backbone (Qwen2Model)
        # CosyVoice2 -> self.llm (Qwen2Encoder) -> self.model (Qwen2ForCausalLM) -> self.model (Qwen2Model)
        if hasattr(self.base_llm, "model"):
            llm_model = self.base_llm.model
            if hasattr(llm_model, "model"):
                self.transformer = llm_model.model
            else:
                self.transformer = llm_model
        else:
            self.transformer = self.base_llm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels=None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Modified forward pass to handle the split embedding structure based on:
        - [0~152703]: Text tokens (transformer.embed_tokens)
        - [152704~152705]: Special tokens (llm_embedding)
        - [152706~159270]: Speech tokens (speech_embedding / llm_decoder)
        """
        # 1. Routing embeddings
        if self.llm_embedding is not None and self.speech_embedding is not None:
            # Safely get hidden size and dtype
            hidden_size = getattr(self.teacher.config, "hidden_size", 896)
            dtype = getattr(self.teacher, "dtype", torch.float32)
            if not isinstance(
                dtype, torch.dtype
            ):  # Handle cases where .dtype might be a property
                dtype = next(self.teacher.parameters()).dtype

            inputs_embeds = torch.zeros(
                (input_ids.size(0), input_ids.size(1), hidden_size),
                device=input_ids.device,
                dtype=dtype,
            )

            # Mask for indices
            is_text = input_ids < self.special_token_offset
            is_special = (input_ids >= self.special_token_offset) & (
                input_ids < self.speech_token_offset
            )
            is_speech = input_ids >= self.speech_token_offset

            # Fill text embeddings (standard vocab)
            if is_text.any():
                text_ids = input_ids.clone()
                text_ids[~is_text] = 0
                # text embeddings are in transformer.embed_tokens
                inputs_embeds[is_text] = self.transformer.embed_tokens(text_ids)[
                    is_text
                ]

            # Fill special tokens (llm_embedding: index 0 and 1)
            if is_special.any():
                special_ids = input_ids - self.special_token_offset
                special_ids[~is_special] = 0
                inputs_embeds[is_special] = self.llm_embedding(special_ids)[is_special]

            # Fill speech embeddings (speech_embedding: index 0 to 6563)
            if is_speech.any():
                speech_ids = input_ids - self.speech_token_offset
                speech_ids[~is_speech] = 0
                inputs_embeds[is_speech] = self.speech_embedding(speech_ids)[is_speech]

            # 2. Call transformer backbone
            # Explicitly ensure attention_mask is a boolean mask as expected by the
            # teacher's internal logic (modeling.py logic: masks = ~make_pad_mask)
            if attention_mask is not None:
                # Convert 1/0 mask to boolean if it's not already
                if not attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.bool()

            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
        else:
            # Fallback for standard models
            outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        # 2. Align output logits (Merge Text + Speech)
        if self.llm_decoder is not None:
            # Extract last hidden states from the transformer output
            if hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                last_hidden = outputs.hidden_states[-1]
            else:
                last_hidden = (
                    outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                )

            # Initialize unified logits mapping to the student's vocab space
            vocab_size = getattr(self.teacher.config, "vocab_size", 159271)
            full_logits = torch.full(
                (last_hidden.size(0), last_hidden.size(1), vocab_size),
                -10000.0,
                device=last_hidden.device,
                dtype=last_hidden.dtype,
            )

            # A. Fill Text Logits (from original teacher lm_head)
            if hasattr(self.base_llm, "model") and hasattr(
                self.base_llm.model, "lm_head"
            ):
                text_logits = self.base_llm.model.lm_head(last_hidden)
                num_text_tokens = min(text_logits.size(-1), self.text_vocab_size)
                full_logits[..., :num_text_tokens] = text_logits[..., :num_text_tokens]

            # B. Fill Speech Logits (from specialized llm_decoder)
            speech_logits = self.llm_decoder(last_hidden)
            num_speech_tokens = min(speech_logits.size(-1), 6564)
            start_idx = self.speech_token_offset
            end_idx = start_idx + num_speech_tokens
            if end_idx <= vocab_size:
                full_logits[..., start_idx:end_idx] = speech_logits[
                    ..., :num_speech_tokens
                ]

            # C. Compute Teacher Loss if labels provided
            teacher_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = full_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                teacher_loss = loss_fct(
                    shift_logits.view(-1, vocab_size), shift_labels.view(-1)
                )

            # Return standard CausalLMOutput
            return CausalLMOutputWithPast(
                loss=teacher_loss,
                logits=full_logits,
                hidden_states=getattr(outputs, "hidden_states", None),
                past_key_values=getattr(outputs, "past_key_values", None),
            )

        return outputs

    @property
    def config(self):
        return self.teacher.config

    @property
    def device(self):
        return next(self.parameters()).device


class CosyVoiceTokenizerWrapper:
    """
    Wrapper for the text tokenizer that handles CosyVoice2 specific tokens:
    - Text: Standard Qwen2 tokenization
    - SOS: <|sos|> or <|text_start|> -> offset
    - Task: <|sft_text_only|> or <|semantic_token_start|> -> offset + 1
    - Speech: <|idx|> -> offset + 2 + idx
    - EOS: <|semantic_token_end|> -> offset + 2 + 6561
    """

    def __init__(self, tokenizer, text_vocab_size=152704):
        import re

        self.tokenizer = tokenizer
        self.text_vocab_size = text_vocab_size
        self.sos_token_id = text_vocab_size
        self.task_token_id = text_vocab_size + 1
        self.speech_token_offset = text_vocab_size + 2
        self.speech_eos_id = self.speech_token_offset + 6561

        # Updated mapping based on data.py and COSYVOICE2.md
        self.special_map = {
            "<|sos|>": self.sos_token_id,
            "<|text_start|>": self.sos_token_id,
            "<|sft_text_only|>": self.task_token_id,
            "<|semantic_token_start|>": self.task_token_id,
            "<|semantic_token_end|>": self.speech_eos_id,
        }

        # Pattern to catch all special markers
        keys_pattern = "|".join(re.escape(k) for k in self.special_map.keys())
        self.pattern = re.compile(rf"({keys_pattern}|<\|(\d+)\|>)")

    def encode(self, text, add_special_tokens=False, **kwargs):
        if not isinstance(text, str):
            # Fallback for non-string inputs (like pre-tokenized)
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

        tokens = []
        last_end = 0
        for match in self.pattern.finditer(text):
            # Process normal text segment
            segment = text[last_end : match.start()]
            if segment:
                # Use base tokenizer for actual text
                tokens.extend(self.tokenizer.encode(segment, add_special_tokens=False))

            # Process special token match
            full_match = match.group(1)
            if full_match in self.special_map:
                tokens.append(self.special_map[full_match])
            elif match.group(2) is not None:  # <|(\d+)|>
                speech_idx = int(match.group(2))
                tokens.append(self.speech_token_offset + speech_idx)

            last_end = match.end()

        # Final segment
        segment = text[last_end:]
        if segment:
            tokens.extend(self.tokenizer.encode(segment, add_special_tokens=False))

        return tokens

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_attention_mask=True,
        **kwargs
    ):
        # Handle single vs batch
        is_batch = isinstance(text, (list, tuple))
        texts = text if is_batch else [text]

        all_ids = [self.encode(t, **kwargs) for t in texts]

        # Padding logic
        if padding or return_tensors == "pt":
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)

            pad_id = getattr(self.tokenizer, "pad_token_id", 0)
            if pad_id is None:
                pad_id = 0

            padded_ids = []
            attention_masks = []
            for ids in all_ids:
                if truncation and len(ids) > max_len:
                    ids = ids[:max_len]
                p_len = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * p_len)
                attention_masks.append([1] * len(ids) + [0] * p_len)
        else:
            padded_ids = all_ids
            attention_masks = [[1] * len(ids) for ids in all_ids]

        # Format output
        res = {"input_ids": padded_ids}
        if return_attention_mask:
            res["attention_mask"] = attention_masks

        if return_tensors == "pt":
            res["input_ids"] = torch.tensor(res["input_ids"], dtype=torch.long)
            if "attention_mask" in res:
                res["attention_mask"] = torch.tensor(res["attention_mask"], dtype=torch.long)

        # Unpack if not batch
        if not is_batch:
            res = {k: v[0] if return_tensors != "pt" else v for k, v in res.items()}

        return res

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
