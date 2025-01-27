import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import highpass_biquad, lowpass_biquad

import whisper
from jiwer import wer
import noisereduce as nr
from demucs import pretrained
from demucs.apply import apply_model


##############################################################################
# 1. Зареждане/запис на вече съществуващи транскрипции (в JSON)
##############################################################################
def load_transcriptions_json(json_path):
    """
    Опитва се да прочете JSON файл със структура:
      {
        "име-на-файл.wav": "транскрипция..."
        ...
      }
    Ако файлът не съществува, връща празен речник.
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[WARNING] Неуспешно четене на {json_path}: {e}")
        return {}

def save_transcriptions_json(transcriptions_dict, json_path):
    """
    Записва речника transcriptions_dict в JSON файл, с utf-8 и отстъп.
    """
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcriptions_dict, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Записани транскрипции в {json_path}")
    except Exception as e:
        print(f"[ERROR] Неуспешен запис на транскрипции: {e}")


##############################################################################
# 2. Основни функции за зареждане, транскрипция, добавяне на шум, денойз
##############################################################################
def load_clean_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        transform = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = transform(waveform)
        sr = target_sr
    return waveform, sr

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

# Зареждаме HTDemucs (ако ще го ползваме)
demucs_model = pretrained.get_model(name='htdemucs')

def bandpass_filter(waveform, sr, low_freq=300, high_freq=3400):
    waveform = highpass_biquad(waveform, sr, low_freq)
    waveform = lowpass_biquad(waveform, sr, high_freq)
    return waveform

def segment_audio(audio, segment_length=16000):
    num_segments = audio.size(1) // segment_length
    segments = torch.split(audio, segment_length, dim=1)
    if audio.size(1) % segment_length != 0:
        segments = list(segments) + [audio[:, num_segments * segment_length:]]
    return segments

def adaptive_noise_profile(segment):
    # Оценка на шума въз основа на нискоенергийни части
    threshold = 0.05 * torch.max(segment.abs())
    noise_estimation = segment[segment.abs() < threshold]
    if len(noise_estimation) == 0:
        # fallback: първите 100 ms
        tenth = max(1, segment.size(1) // 10)
        noise_estimation = segment[:, :tenth]
    return noise_estimation

def reduce_noise_segment_adaptive(segment, sr):
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

def denoise_audio_with_noisereduce(noisy_waveform, sr=16000):
    filtered = bandpass_filter(noisy_waveform, sr)
    segments = segment_audio(filtered)
    denoised_segments = [reduce_noise_segment_adaptive(seg, sr) for seg in segments]
    return torch.cat(denoised_segments, dim=1)

def denoise_audio_nr(noisy_waveform, orig_sr=16000):
    return denoise_audio_with_noisereduce(noisy_waveform, orig_sr)

def denoise_audio_demucs(noisy_waveform, orig_sr=16000):
    resample_to_44k = T.Resample(orig_freq=orig_sr, new_freq=44100)
    resample_to_orig = T.Resample(orig_freq=44100, new_freq=orig_sr)

    waveform_44k = resample_to_44k(noisy_waveform)
    waveform_44k = waveform_44k.unsqueeze(0)

    sources = apply_model(demucs_model, waveform_44k)

    # Примерно speech обикновено е [0][0] или [0][1] в зависимост от модела
    clean_speech_44k = sources[0][1]

    clean_speech = resample_to_orig(clean_speech_44k.unsqueeze(0)).squeeze(0)
    return clean_speech


##############################################################################
# 3. Модели за транскрипция (Whisper)
##############################################################################
whisper_model_medium = whisper.load_model("medium")
whisper_model_large = whisper.load_model("large")

def transcribe_audio(waveform, sr=16000, model_type="medium", language='bg', temp_file="temp.wav"):
    """
    Транскрибира в зависимост от зададения model_type ("medium" или "large").
    """
    if model_type == "medium":
        w_model = whisper_model_medium
    elif model_type == "large":
        w_model = whisper_model_large
    else:
        raise ValueError("Unknown model type")

    torchaudio.save(temp_file, waveform, sr)
    result = w_model.transcribe(temp_file, language=language)
    os.remove(temp_file)
    return result["text"]

def evaluate_denoising(ref_text, noisy_text, denoised_text):
    wer_noisy = wer(ref_text, noisy_text)
    wer_denoised = wer(ref_text, denoised_text)
    return wer_noisy, wer_denoised


##############################################################################
# 4. Основна функция:
#    - Чете/попълва JSON с "чисти" транскрипции
#    - Добавя шум, денойзва, сравнява
##############################################################################
def main(
    clean_audio_folder="clean_audio",
    noise_audio_path="white-noise.mp3",
    transcriptions_json="clean_transcriptions.json"
):
    # 4.1 Зареждаме речника с вече известни транскрипции
    clean_transcriptions = load_transcriptions_json(transcriptions_json)
    print(f"[INFO] Заредени {len(clean_transcriptions)} транскрипции от {transcriptions_json}")

    # 4.2 Транскрибиране на нови файлове (ако не фигурират в речника)
    file_list = os.listdir(clean_audio_folder)
    for file_name in file_list:
        if file_name not in clean_transcriptions:
            file_path = os.path.join(clean_audio_folder, file_name)
            # Зареждаме аудиото
            clean_audio, sr = load_clean_audio(file_path)
            # Транскрибираме, примерно с 'large'
            text = transcribe_audio(clean_audio, sr, model_type="large", language="bg", temp_file="temp_clean.wav")
            clean_transcriptions[file_name] = text
            print(f"[INFO] Нов файл '{file_name}' => транскрибиран.")

    # 4.3 Записваме обновения речник обратно
    save_transcriptions_json(clean_transcriptions, transcriptions_json)

    # 4.4 Зареждаме аудио за шум (еднократно)
    noise_audio, _ = load_clean_audio(noise_audio_path)

    # 4.5 Сравнение - за всеки файл добавяме шум, денойзваме, смятаме WER
    count_better_denoised = 0
    count_better_noisy = 0

    for file_name in file_list:
        # Ако няма транскрипция (примерно файлът не е аудио), пропускаме
        if file_name not in clean_transcriptions:
            continue

        ref_text = clean_transcriptions[file_name]
        file_path = os.path.join(clean_audio_folder, file_name)
        clean_audio, sr = load_clean_audio(file_path)

        # Добавяме шум
        noisy_audio = add_noise(clean_audio, noise_audio, snr_db=10)
        # Денойзваме (noisereduce или demucs)
        denoised_audio = denoise_audio_nr(noisy_audio, orig_sr=sr)
        # denoised_audio = denoise_audio_demucs(noisy_audio, orig_sr=sr)

        # Транскрибиране (примерно с "medium")
        noisy_text = transcribe_audio(noisy_audio, sr, model_type="medium", language="bg", temp_file="temp_noisy.wav")
        denoised_text = transcribe_audio(denoised_audio, sr, model_type="medium", language="bg", temp_file="temp_denoised.wav")

        # WER сравнение
        wer_noisy, wer_denoised = evaluate_denoising(ref_text, noisy_text, denoised_text)

        print(f"\n[Файл: {file_name}]")
        print(f"  * Чиста транскрипция: {ref_text}")
        print(f"  * Noisy  транскрипция: {noisy_text}")
        print(f"  * Denoised транскрипция: {denoised_text}")
        print(f"  * WER(Noisy) = {wer_noisy:.3f}")
        print(f"  * WER(Denoised) = {wer_denoised:.3f}")

        if wer_noisy > wer_denoised:
            count_better_denoised += 1
        else:
            count_better_noisy += 1

    print("\n=== Обобщение ===")
    print(f"Брой случаи, в които денойзнатото има по-нисък WER: {count_better_denoised}")
    print(f"Брой случаи, в които шумното има по-нисък или равен WER: {count_better_noisy}")


if __name__ == "__main__":
    main()
