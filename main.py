
import torchaudio
import torchaudio.transforms as T
import torch
import librosa
import numpy as np
import whisper
from demucs import pretrained
from demucs.apply import apply_model
from jiwer import wer
import os
import speechbrain as sb
import noisereduce as nr
# import speechbrain as sb
# from speechbrain.inference.separation import SepformerSeparation
import torch
from torchaudio.functional import highpass_biquad, lowpass_biquad


# 1. Зареждане на данни (пример с Common Voice)
def load_clean_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    transform = T.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = transform(waveform)
    return waveform, target_sr


# 2. Добавяне на шум към речта (подравняване на дължините)
def add_noise(clean_waveform, noise_waveform, snr_db=10):
    clean_len = clean_waveform.shape[1]
    noise_len = noise_waveform.shape[1]

    if noise_len < clean_len:
        repeats = (clean_len // noise_len) + 1
        noise_waveform = noise_waveform.repeat(1, repeats)[:, :clean_len]
    else:
        noise_waveform = noise_waveform[:, :clean_len]

    noise_rms = torch.sqrt(torch.mean(noise_waveform ** 2))
    clean_rms = torch.sqrt(torch.mean(clean_waveform ** 2))
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise_waveform = noise_waveform * (desired_noise_rms / noise_rms)
    return clean_waveform + noise_waveform


# 3. Премахване на шум с Demucs
model = pretrained.get_model(name='htdemucs')

# Зареждане на предварително обучен модел за премахване на шум
# separator = SepformerSeparation.from_hparams(source="speechbrain/sepformer-whamr-v1", savedir="models")

# Функция за премахване на шум с Noisereduce
# Bandpass filter to focus on speech frequencies (300–3400 Hz)
def bandpass_filter(waveform, sr, low_freq=300, high_freq=3400):
    waveform = highpass_biquad(waveform, sr, low_freq)
    waveform = lowpass_biquad(waveform, sr, high_freq)
    return waveform


# Split audio into smaller segments for better noise reduction
def segment_audio(audio, segment_length=16000):  # 1 second at 16kHz
    num_segments = audio.size(1) // segment_length
    segments = torch.split(audio, segment_length, dim=1)
    if audio.size(1) % segment_length != 0:
        segments = list(segments) + [audio[:, num_segments * segment_length:]]
    return segments


def adaptive_noise_profile(segment):
    # Identify low-energy parts of the segment as noise
    noise_estimation = segment[segment.abs() < (0.05 * torch.max(segment.abs()))]
    if len(noise_estimation) == 0:
        # Fallback: Use the first 100 ms as noise
        noise_estimation = segment[:, :segment.size(1) // 10]
    return noise_estimation


def reduce_noise_segment_adaptive(segment, sr):
    # Estimate adaptive noise profile
    noise_profile = adaptive_noise_profile(segment).numpy()
    segment_np = segment.numpy()
    denoised_segment = nr.reduce_noise(
        y=segment_np,
        sr=sr,
        y_noise=noise_profile,
        n_fft=1024,
        hop_length=256,
        stationary=False
    )
    return torch.tensor(denoised_segment).float()


# Apply Noisereduce with advanced parameters and noise profile
def reduce_noise_segment(segment, sr, noise_profile=None):
    segment_np = segment.numpy()
    if noise_profile is not None:
        noise_profile = noise_profile.numpy()
    denoised_segment = nr.reduce_noise(
        y=segment_np,
        sr=sr,
        y_noise=noise_profile,
        n_fft=1024,  # Higher frequency resolution
        hop_length=256,  # Overlap between frames
        stationary=False  # Non-stationary noise handling
    )
    return torch.tensor(denoised_segment).float()


# Full pipeline for denoising audio
def denoise_audio_with_noisereduce(noisy_waveform, sr=16000):
    # Step 1: Bandpass filter to focus on speech frequencies
    filtered_waveform = bandpass_filter(noisy_waveform, sr)

    # Step 2: Extract noise profile (first second or external noise sample)
    noise_profile = filtered_waveform[:, :sr]  # First second of the audio

    # Step 3: Split the waveform into smaller segments
    segments = segment_audio(filtered_waveform)

    # Step 4: Apply noise reduction to each segment
    denoised_segments = [
        reduce_noise_segment_adaptive(segment, sr) for segment in segments
    ]

    # Step 5: Reconstruct the full waveform
    denoised_waveform = torch.cat(denoised_segments, dim=1)

    return denoised_waveform


# Промяна на функцията `denoise_audio` да използва Noisereduce
def denoise_audio_nr(noisy_waveform, orig_sr=16000):
    denoised_speech = denoise_audio_with_noisereduce(noisy_waveform, orig_sr)
    return denoised_speech


# Останалата част от кода остава същата


def denoise_audio_demucs(noisy_waveform, orig_sr=16000):
    # Resample to 44.1 kHz, which Demucs requires
    resample_to_44k = T.Resample(orig_freq=orig_sr, new_freq=44100)
    resample_to_orig = T.Resample(orig_freq=44100, new_freq=orig_sr)

    noisy_waveform_44k = resample_to_44k(noisy_waveform)

    # Demucs expects waveform as a 3D tensor (batch, channels, samples)
    noisy_waveform_44k = noisy_waveform_44k.unsqueeze(0)

    # Apply the Demucs model
    sources = apply_model(model, noisy_waveform_44k)

    # Extract clean speech (assumes first source is speech)
    clean_speech_44k = sources[0][1]

    # Resample back to the original sampling rate
    clean_speech = resample_to_orig(clean_speech_44k.unsqueeze(0)).squeeze(0)
    return clean_speech


# 4. Транскрипция с Whisper
whisper_model = whisper.load_model("medium")
whisper_model_large = whisper.load_model("large")


def transcribe_audio(waveform, sr=16000):
    temp_file = "temp.wav"
    torchaudio.save(temp_file, waveform, sr)
    result = whisper_model.transcribe(temp_file, language='bg')
    os.remove(temp_file)
    return result["text"]


def transcribe_audio_large(waveform, sr=16000):
    temp_file = "temp.wav"
    torchaudio.save(temp_file, waveform, sr)
    result = whisper_model_large.transcribe(temp_file, language='bg')
    os.remove(temp_file)
    return result["text"]

# 5. Изчисляване на WER
def evaluate_denoising(clean_text, noisy_text, denoised_text):
    wer_noisy = wer(clean_text, noisy_text)
    wer_denoised = wer(clean_text, denoised_text)
    return wer_noisy, wer_denoised


# 6. Основна функция
def main(clean_audio_folder, noise_audio_path):
    noise_audio, _ = load_clean_audio(noise_audio_path)
    denoised = 0
    noisy = 0
    for file_name in os.listdir(clean_audio_folder):
        file_path = os.path.join(clean_audio_folder, file_name)
        clean_audio, sr = load_clean_audio(file_path)

        noisy_audio = add_noise(clean_audio, noise_audio)
        denoised_audio = denoise_audio_nr(noisy_audio, orig_sr=sr)
        torchaudio.save("noise_" + file_name, noisy_audio, 16000)
        torchaudio.save("denoised_" + file_name, denoised_audio, 16000)

        clean_text = transcribe_audio_large(clean_audio, sr)
        print(clean_text)
        noisy_text = transcribe_audio(noisy_audio, sr)
        print(noisy_text)
        denoised_text = transcribe_audio(denoised_audio, sr)
        print(denoised_text)

        wer_noisy, wer_denoised = evaluate_denoising(clean_text, noisy_text, denoised_text)

        print(f"Файл: {file_name}")
        print(f"WER на шумен запис: {wer_noisy}")
        print(f"WER след изчистване: {wer_denoised}")

        if wer_noisy > wer_denoised:
            denoised += 1
        else:
            noisy += 1

    print(f"шумен запис: {noisy}")
    print(f"след изчистване: {denoised}")


if __name__ == "__main__":
    clean_audio_folder = "clean_audio"
    noise_audio_path = "white-noise.mp3"
    main(clean_audio_folder, noise_audio_path)

'''
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

def split_audio(file_path, output_dir, chunk_length_ms=10000):
    """
    Splits an audio file into chunks of specified length.

    :param file_path: Path to the input audio file
    :param output_dir: Directory to save the chunks
    :param chunk_length_ms: Length of each chunk in milliseconds (default 10 seconds)
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split audio into chunks
    chunks = make_chunks(audio, chunk_length_ms)

    # Export each chunk
    for i, chunk in enumerate(chunks):
        chunk_name = os.path.join(output_dir, f"chunk_{i+1}.wav")
        chunk.export(chunk_name, format="wav")
        print(f"Exported {chunk_name}")

# Example usage
if __name__ == "__main__":
    input_file = "BNR-news-2025-01-22-12-00.mp3"  # Replace with your audio file path
    output_directory = "output_chunks"  # Replace with your desired output directory

    split_audio(input_file, output_directory)
'''
