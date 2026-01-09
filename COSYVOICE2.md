## Architecture

CosyVoice2 is built upon the Qwen2 architecture, which is a transformer-based language model. The model consists of an embedding layer, multiple transformer decoder layers, and a final linear layer for output generation. The architecture is designed to handle both text and speech embeddings, allowing it to perform text-to-speech synthesis task effectively.

```
Qwen2LM(
  (llm_embedding): Embedding(2, 896)
  (llm): Qwen2Encoder(
    (model): Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(153024, 896)
        (layers): ModuleList(
          (0-23): 24 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): Linear(in_features=896, out_features=896, bias=True)
              (k_proj): Linear(in_features=896, out_features=128, bias=True)
              (v_proj): Linear(in_features=896, out_features=128, bias=True)
              (o_proj): Linear(in_features=896, out_features=896, bias=False)
            )
            (mlp): Qwen2MLP(
              (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
              (up_proj): Linear(in_features=896, out_features=4864, bias=False)
              (down_proj): Linear(in_features=4864, out_features=896, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((896,), eps=1e-06)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (lm_head): Linear(in_features=896, out_features=153024, bias=False)
    )
  )
  (llm_decoder): Linear(in_features=896, out_features=6564, bias=True)
  (criterion_ce): LabelSmoothingLoss(
    (criterion): KLDivLoss()
  )
  (speech_embedding): Embedding(6564, 896)
)
```

## Input Representation

### Speech Token Size

Total speech tokens: 6561, there are 3 special tokens

### Special Tokens

```python
...
self.sos = 0
self.task_id = 1
self.eos_token = speech_token_size
self.fill_token = speech_token_size + 2
...
```

### Pad Token

Pad token is self.speech_token_size + 1 = 654 + 1 = 655.

Reference:

```python
acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
```

llm_embedding is two special tokens for TTS tasks, 0 for sos and 1 for task_id.

eos is self.speech_token_size + 1, which is 654 + 1 = 6564.

### Input Sequence

The input to the model consists of a sequence that starts with a start-of-sequence (sos) token embedding, followed by text embeddings, a task identifier embedding, and finally speech embeddings. The input sequence is padded to ensure uniform length across the batch.

```python
 def pad_unpad_sequence(self, sos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len
```

The input format:

```[sos_emb] + [text_embedding_1, text_embedding_2, ..., text_embedding_n] + [task_id_emb] + [speech_embedding_1, speech_embedding_2, ..., speech_embedding_m] + [eos]```