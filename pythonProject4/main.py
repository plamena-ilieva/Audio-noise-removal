import torch
import os
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.ASR import EncoderDecoderASR
from jiwer import wer
from shutil import copytree

# 1. Load audio data
def load_audio(file_path, target_sr=16000):
    """Load audio file and resample to the target sample rate."""
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr

# 2. Add noise to audio
def add_noise(clean_waveform, noise_waveform, snr_db=10):
    """Add noise to clean audio at a given SNR (Signal-to-Noise Ratio)."""
    clean_len = clean_waveform.shape[1]
    noise_len = noise_waveform.shape[1]

    if noise_len < clean_len:
        repeats = (clean_len // noise_len) + 1
        noise_waveform = noise_waveform.repeat(1, repeats)[:, :clean_len]
    else:
        noise_waveform = noise_waveform[:, :clean_len]

    clean_rms = torch.sqrt(torch.mean(clean_waveform ** 2))
    noise_rms = torch.sqrt(torch.mean(noise_waveform ** 2))
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_waveform = noise_waveform * (desired_noise_rms / noise_rms)
    return clean_waveform + noise_waveform

# 3. Preprocess waveform for model input
def preprocess_waveform(waveform):
    """Ensure waveform has shape [Batch, Channels, Samples] and is mono."""
    # If the waveform has multiple channels, convert to mono
    if waveform.size(0) > 1:  # Multiple channels (e.g., stereo)
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Average across channels

    # Ensure the waveform has [Batch, Channels, Samples]
    if len(waveform.shape) == 2:  # [Channels, Samples]
        waveform = waveform.unsqueeze(0)  # Add batch dimension: [1, Channels, Samples]
    elif len(waveform.shape) == 1:  # [Samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel: [1, 1, Samples]
    elif len(waveform.shape) == 4:  # [Batch, 1, 1, Samples]
        waveform = waveform.squeeze(1)  # Remove unnecessary dimension

    return waveform


# 4. Initialize Sepformer model
def initialize_sepformer(savedir="models/sepformer"):
    """Initialize SpeechBrain's Sepformer model with symlink privilege workaround."""
    try:
        return SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr", savedir=savedir)
    except OSError as e:
        if "WinError 1314" in str(e):
            print("Symlink privilege error detected. Copying files instead of linking...")
            src_path = os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--sepformer-whamr/snapshots")
            latest_snapshot = sorted(os.listdir(src_path))[-1]
            copytree(os.path.join(src_path, latest_snapshot), savedir, dirs_exist_ok=True)
            return SepformerSeparation.from_hparams(source=None, savedir=savedir)
        else:
            raise e
    except FileNotFoundError as e:
        print(f"Model files not found. Ensure that the model exists in the path: {savedir}")
        raise e

separator = None
try:
    separator = initialize_sepformer()
except Exception as e:
    print("Error initializing Sepformer model:", e)

# 5. Denoise audio with SpeechBrain Sepformer
def denoise_audio_speechbrain(noisy_waveform, sr=16000):
    """Apply SpeechBrain's Sepformer to denoise audio."""
    if separator is None:
        raise RuntimeError("Sepformer model is not initialized.")

    # Print the shape before processing
    print("Shape before preprocessing:", noisy_waveform.shape)

    # Preprocess the waveform to ensure correct shape
    # noisy_waveform = preprocess_waveform(noisy_waveform)

    # Print the shape after preprocessing
    print("Shape after preprocessing:", noisy_waveform.shape)

    # Pass the waveform to the model
    separated_sources = separator.separate_batch(noisy_waveform)

    # Extract clean waveform
    clean_waveform = separated_sources[0]  # Take the first source (speech)

    return clean_waveform.T


# 6. Initialize ASR model
def initialize_asr_model(savedir="models/asr_model"):
    """Initialize SpeechBrain's ASR model with symlink privilege workaround."""
    try:
        return EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir=savedir)
    except OSError as e:
        if "WinError 1314" in str(e):
            print("Symlink privilege error detected. Copying files instead of linking...")
            src_path = os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots")
            latest_snapshot = sorted(os.listdir(src_path))[-1]
            copytree(os.path.join(src_path, latest_snapshot), savedir, dirs_exist_ok=True)
            return EncoderDecoderASR.from_hparams(source=None, savedir=savedir)
        else:
            raise e
    except FileNotFoundError as e:
        print(f"ASR model files not found. Ensure that the model exists in the path: {savedir}")
        raise e

asr_model = None
try:
    asr_model = initialize_asr_model()
except Exception as e:
    print("Error initializing ASR model:", e)

# 7. Transcribe audio with SpeechBrain ASR
def transcribe_audio(waveform, sr=16000):
    """Transcribe audio to text using SpeechBrain's ASR."""
    if asr_model is None:
        raise RuntimeError("ASR model is not initialized.")

    # Preprocess waveform for ASR input
    if len(waveform.shape) == 1:  # If waveform is [Samples]
        waveform = waveform.unsqueeze(0)  # Add batch dimension: [1, Samples]
    elif len(waveform.shape) == 2 and waveform.shape[0] > 1:  # Stereo case
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

    # Ensure waveform has [Batch, Samples]
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dimension if missing: [1, 1, Samples]

    # Calculate wav_lens (relative lengths)
    wav_lens = torch.tensor([1.0])  # Single audio sample with full length

    # Transcribe
    transcription = asr_model.transcribe_batch(waveform, wav_lens)
    return transcription[0]


# 8. Evaluate Word Error Rate (WER)
def evaluate_denoising(clean_text, noisy_text, denoised_text):
    """Evaluate WER (Word Error Rate) for noisy and denoised audio."""
    wer_noisy = wer(clean_text, noisy_text)
    wer_denoised = wer(clean_text, denoised_text)
    return wer_noisy, wer_denoised

# 9. Main function
def main(clean_audio_folder, noise_audio_path):
    noise_audio, _ = load_audio(noise_audio_path)

    for file_name in os.listdir(clean_audio_folder):
        file_path = os.path.join(clean_audio_folder, file_name)
        clean_audio, sr = load_audio(file_path)

        noisy_audio = add_noise(clean_audio, noise_audio)
        torchaudio.save("nois.wav", noisy_audio, sr)

        try:
            denoised_audio = denoise_audio_speechbrain(noisy_audio, sr)
            torchaudio.save("denois.wav", denoised_audio, sr)
        except RuntimeError as e:
            print(f"Skipping denoising for {file_name}: {e}")
            continue

        try:
            clean_text = transcribe_audio(clean_audio, sr)
            noisy_text = transcribe_audio(noisy_audio, sr)
            denoised_text = transcribe_audio(denoised_audio, sr)
        except RuntimeError as e:
            print(f"Skipping transcription for {file_name}: {e}")
            continue

        wer_noisy, wer_denoised = evaluate_denoising(clean_text, noisy_text, denoised_text)

        print(f"File: {file_name}")
        print(f"Clean transcription: {clean_text}")
        print(f"Noisy transcription: {noisy_text}")
        print(f"Denoised transcription: {denoised_text}")
        print(f"WER (noisy): {wer_noisy:.2f}")
        print(f"WER (denoised): {wer_denoised:.2f}")

if __name__ == "__main__":
    clean_audio_folder = "clean_audio"
    noise_audio_path = "noise-drum.mp3"
    main(clean_audio_folder, noise_audio_path)
