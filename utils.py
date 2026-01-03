import s3tokenizer
import numpy as np
import torch
import torchaudio
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from transformers import AutoTokenizer

speech_tokenizer = None
TARGET_SAMPLING_RATE = 16000


def _resample_audio(audio: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
    """
    Resample audio to target sampling rate (16000 Hz) if needed using torchaudio.

    Args:
        audio: Audio samples as numpy array or torch tensor
        sr: Current sampling rate

    Returns:
        Resampled audio at 16000 Hz as torch tensor
    """
    if sr == TARGET_SAMPLING_RATE:
        # No resampling needed, convert to tensor if numpy
        if isinstance(audio, np.ndarray):
            return torch.from_numpy(audio).float()
        return audio.float()

    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float()

    # Add batch dimension if needed
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Resample using torchaudio
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=TARGET_SAMPLING_RATE
    )
    resampled = resampler(audio_tensor)

    # Remove batch dimension and return as tensor
    return resampled.squeeze(0)


def get_speech_tokens(audio_input: Union[str, np.ndarray, dict], device="cpu"):
    """
    Convert audio to speech tokens.

    Args:
        audio_input: Either:
            - File path (str)
            - Numpy array of audio samples
            - Dict with 'array' and 'sampling_rate' keys (HuggingFace format)
        device: Device to use for processing (default: "cpu")

    Returns:
        codes: Speech token codes
        codes_len: Length of codes
    """
    global speech_tokenizer
    if speech_tokenizer is None:
        print("Loading speech tokenizer...")
        speech_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").to(device)

    # Handle different input formats
    if isinstance(audio_input, dict):
        # HuggingFace dataset format with 'array' and 'sampling_rate'
        audio = audio_input.get("array")
        sr = audio_input.get("sampling_rate", TARGET_SAMPLING_RATE)
    elif isinstance(audio_input, str):
        # File path
        audio = s3tokenizer.load_audio(audio_input)
        sr = TARGET_SAMPLING_RATE  # Assume s3tokenizer loads at target rate
    else:
        # Numpy array (assume it's at target sampling rate)
        audio = audio_input
        sr = TARGET_SAMPLING_RATE

    # Resample if necessary (returns tensor)
    audio_tensor = _resample_audio(audio, sr).to(device)

    log_mel = s3tokenizer.log_mel_spectrogram(audio_tensor)
    mels, mels_lens = s3tokenizer.padding([log_mel])

    # Move mel spectrogram and lengths to the correct device for the speech tokenizer
    mels = mels.to(device)
    mels_lens = mels_lens.to(device)

    codes, codes_len = speech_tokenizer.quantize(mels, mels_lens)
    codes = codes.to(device)
    codes_len = codes_len.to(device)

    return codes, codes_len


def get_speech_tokens_batch(audio_inputs: list, device="cpu"):
    """
    Convert a batch of audio inputs to speech tokens.

    Args:
        audio_inputs: List of audio inputs (str, np.ndarray, or dict)
        device: Device to use for processing (default: "cpu")

    Returns:
        codes: Batch of speech token codes
        codes_len: Batch of lengths of codes
    """
    global speech_tokenizer
    if speech_tokenizer is None:
        print("Loading speech tokenizer...")
        speech_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").to(device)

    log_mels = []
    for audio_input in audio_inputs:
        # Handle different input formats
        if isinstance(audio_input, dict):
            audio = audio_input.get("array")
            sr = audio_input.get("sampling_rate", TARGET_SAMPLING_RATE)
        elif isinstance(audio_input, str):
            audio = s3tokenizer.load_audio(audio_input)
            sr = TARGET_SAMPLING_RATE
        else:
            audio = audio_input
            sr = TARGET_SAMPLING_RATE

        # Resample if necessary (returns tensor)
        audio_tensor = _resample_audio(audio, sr).to(device)
        log_mel = s3tokenizer.log_mel_spectrogram(audio_tensor)
        log_mels.append(log_mel)

    mels, mels_lens = s3tokenizer.padding(log_mels)

    # Move mel spectrogram and lengths to the correct device
    mels = mels.to(device)
    mels_lens = mels_lens.to(device)

    codes, codes_len = speech_tokenizer.quantize(mels, mels_lens)
    codes = codes.to(device)
    codes_len = codes_len.to(device)

    return codes, codes_len


def prepare_inputs(
    text: str,
    audio_input: Union[str, np.ndarray],
    prefix: str,
    text_bos: str,
    text_eos: str,
    text_prefix: str,
    speech_bos: str,
    speech_eos: str,
    tokenizer: "AutoTokenizer",
    device="cpu",
):
    """
    Prepare model inputs with text and speech tokens.

    Args:
        text: Text input
        audio_input: Either a file path (str) or numpy array of audio samples
        prefix: Prefix to add before text
        text_bos: Text beginning-of-sequence token
        text_eos: Text end-of-sequence token
        text_prefix: Prefix to add after text (before speech tokens)
        speech_bos: Speech beginning-of-sequence token
        speech_eos: Speech end-of-sequence token
        tokenizer: Tokenizer for text encoding
        device: Device to use (default: "cpu")

    Returns:
        Tokenized inputs with input_ids and attention_mask
    """
    codes, _ = get_speech_tokens(audio_input, device=device)
    speech_tokens = "".join(
        ["<|" + str(token_id) + "|>" for token_id in codes[0].cpu().tolist()]
    )
    text = (
        prefix
        + text_bos
        + text.strip()
        + text_eos
        + text_prefix
        + speech_bos
        + speech_tokens
        + speech_eos
    )

    # Tokenize text
    text_inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)

    return text_inputs


def prepare_inputs_batch(
    texts: list,
    audio_inputs: list,
    prefixes: list,
    text_bos: str,
    text_eos: str,
    text_prefixes: list,
    speech_bos: str,
    speech_eos: str,
    tokenizer: "AutoTokenizer",
    device="cpu",
):
    """
    Prepare a batch of model inputs with text and speech tokens.
    """
    codes, codes_lens = get_speech_tokens_batch(audio_inputs, device=device)

    batch_texts = []
    for i in range(len(texts)):
        speech_tokens = "".join(
            [
                "<|" + str(token_id) + "|>"
                for token_id in codes[i, : codes_lens[i]].cpu().tolist()
            ]
        )
        full_text = (
            prefixes[i]
            + text_bos
            + texts[i].strip()
            + text_eos
            + text_prefixes[i]
            + speech_bos
            + speech_tokens
            + speech_eos
        )
        batch_texts.append(full_text)

    # Tokenize batch
    # We don't pad here because the collator will handle it
    text_inputs = tokenizer(batch_texts, padding=False, return_attention_mask=True)

    return text_inputs
