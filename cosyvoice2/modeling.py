from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
import random
from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoConfig

IGNORE_ID = -1


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrain_path)
        self.model = Qwen2ForCausalLM(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


class CosyVoice2(torch.nn.Module):
    def __init__(
        self,
        model_path,
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        mix_ratio: List[int] = [5, 15],
    ):
        super(CosyVoice2, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2
        self.mix_ratio = mix_ratio

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = Qwen2Encoder(model_path)
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 3, llm_input_size
        )

    def prepare_lm_input_target(
        self,
        sos_emb,
        text_token,
        text_token_emb,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_emb,
        speech_token_len,
        instruct_token=None,
        instruct_token_emb=None,
        instruct_token_len=None,
    ):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        text_token_emb = unpad_sequence(
            text_token_emb, text_token_len.cpu(), batch_first=True
        )
        speech_token_emb = unpad_sequence(
            speech_token_emb, speech_token_len.cpu(), batch_first=True
        )
        # NOTE add instruct_token in CosyVoice3
        if (
            instruct_token is not None
            and instruct_token_emb is not None
            and instruct_token_len is not None
        ):
            instruct_token = unpad_sequence(
                instruct_token, instruct_token_len.cpu(), batch_first=True
            )
            instruct_token_emb = unpad_sequence(
                instruct_token_emb, instruct_token_len.cpu(), batch_first=True
            )
        else:
            instruct_token = [torch.empty(0).to(text_token[0])] * len(text_token)
            instruct_token_emb = [torch.empty(0, 896).to(text_token_emb[0])] * len(
                text_token
            )
            instruct_token_len = torch.zeros(len(text_token)).to(text_token_len)
        for i in range(len(text_token)):
            # bistream sequence
            if (
                random.random() < 0.5
                and speech_token_len[i] / text_token_len[i]
                > self.mix_ratio[1] / self.mix_ratio[0]
            ):
                this_lm_target, this_lm_input = [IGNORE_ID], [sos_emb.squeeze(dim=0)]
                this_lm_target += [IGNORE_ID] * instruct_token_len[i]
                this_lm_input.append(instruct_token_emb[i])
                for j in range(
                    ((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()
                ):
                    this_text_token = text_token[i][
                        j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
                    ].tolist()
                    this_speech_token = speech_token[i][
                        j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
                    ].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.fill_token)
                        this_lm_input.append(
                            text_token_emb[i][
                                j * self.mix_ratio[0] : (j + 1) * self.mix_ratio[0]
                            ]
                        )
                        this_lm_input.append(
                            speech_token_emb[i][
                                j * self.mix_ratio[1] : (j + 1) * self.mix_ratio[1]
                            ]
                        )
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][
                            j * self.mix_ratio[1] :
                        ].tolist()
                        this_lm_target.append(self.eos_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0] :])
                        this_lm_input.append(task_id_emb.squeeze(dim=0))
                        this_lm_input.append(
                            speech_token_emb[i][j * self.mix_ratio[1] :]
                        )
                this_lm_target, this_lm_input = torch.tensor(
                    this_lm_target
                ), torch.concat(this_lm_input, dim=0)
            # unistream sequence
            else:
                this_lm_target = torch.tensor(
                    [IGNORE_ID] * (1 + instruct_token_len[i] + text_token_len[i])
                    + speech_token[i].tolist()
                    + [self.eos_token]
                )
                this_lm_input = torch.concat(
                    [
                        sos_emb.squeeze(dim=0),
                        instruct_token_emb[i],
                        text_token_emb[i],
                        task_id_emb.squeeze(dim=0),
                        speech_token_emb[i],
                    ],
                    dim=0,
                )
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb,
            text_token,
            text_token_emb,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        out = CausalLMOutputWithPast(
            logits=logits,
        )

        return out
