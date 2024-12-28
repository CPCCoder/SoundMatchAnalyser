import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, ifft
import pyloudnorm as pyln
from pesq import pesq
import librosa
import math

def combined_score(lufs, scale, peaq, mse, snr, thd, flatness_diff, centroid_diff, original_flatness, original_centroid):
    lufs = max(lufs, -60)
    norm_lufs = 1.0-(lufs + 60) / 60
    norm_lufs = min(max(norm_lufs, 0.0), 1.0)

    norm_scale = scale
    norm_scale = min(max(norm_scale, 0.0), 1.0)

    norm_peaq = peaq / 4.55
    norm_peaq = min(max(norm_peaq, 0.0), 1.0)

    norm_mse = 1 - mse / 0.01
    norm_mse = min(max(norm_mse, 0.0), 1.0)

    snr = min(snr, 40)
    norm_snr = snr / 40
    norm_snr = min(norm_snr, 1.0)

    norm_thd = 1 - thd / 10

    norm_flatness = 1 - math.pow(flatness_diff / 0.001, 2)
    norm_flatness = min(max(norm_flatness, 0.0), 1.0)

    norm_centroid = 1 - math.pow(centroid_diff / 2000, 2)
    norm_centroid = min(max(norm_centroid, 0.0), 1.0)
    
    weights = {
        "lufs": 0.25,
        "scale": 0.05,
        "peaq": 0.3,
        "mse": 0.08,
        "snr": 0.2,
        "thd": 0.1,
        "flatness": 0.01,
        "centroid": 0.01
    }

    combined_score = (
        weights["lufs"] * norm_lufs +
        weights["scale"] * norm_scale +
        weights["peaq"] * norm_peaq +
        weights["mse"] * norm_mse +
        weights["snr"] * norm_snr +
        weights["thd"] * norm_thd +
        weights["flatness"] * norm_flatness +
        weights["centroid"] * norm_centroid
    )

    combined_score = min(max(combined_score, 0.0), 1.0)

    return combined_score

def fft_bandpass_filter(signal, sr, lowcut, highcut):
    fft_spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1/sr)
    fft_spectrum[(frequencies < lowcut) | (frequencies > highcut)] = 0
    return np.real(np.fft.ifft(fft_spectrum))

def adjust_latency(original, reconstructed, latency):
    if latency > 0:
        reconstructed = np.pad(reconstructed, (latency, 0), mode='constant')[:len(original)]
    elif latency < 0:
        original = np.pad(original, (-latency, 0), mode='constant')[:len(reconstructed)]
    return original, reconstructed

# Function to shift the audio signal by a certain number of samples
def shift_audio(audio, samples):
    if samples > 0:
        return np.pad(audio, (samples, 0), mode='constant')[:len(audio)]
    elif samples < 0:
        return np.pad(audio, (0, -samples), mode='constant')[abs(samples):]
    else:
        return audio

def convert_to_float(data):
    if np.issubdtype(data.dtype, np.integer):
        if np.issubdtype(data.dtype, np.int16):
            return data.astype(np.float32) / 32768.0
        elif np.issubdtype(data.dtype, np.int32):
            return data.astype(np.float32) / 2147483648.0
        else:
            raise ValueError("Unsupported integer data type")
    elif np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    else:
        raise ValueError("Unsupported data type")

# MSE calculation
def calculate_mse(audio1, audio2):
    return np.mean((audio1 - audio2) ** 2)

# LUFS calculation
def calculate_lufs(audio, rate):
    meter = pyln.Meter(rate)  # Create a LUFS meter
    lufs  = meter.integrated_loudness(audio) 
    return lufs

# Coarse and fine latency adjustment
def calc_best_latency(amp, model, rate, max_latency_samples, coarse_step_size=10, fine_step_size=1):
    best_mse = calculate_mse(amp, model)
    best_latency = 0
    
    # Coarse search
    for latency in range(-max_latency_samples, max_latency_samples + 1, coarse_step_size):
        shifted_model = shift_audio(model, latency)
        mse = calculate_mse(amp, shifted_model)
        
        if mse < best_mse:
            best_mse = mse
            best_latency = latency

    # Fine search around the best latency value
    start_latency = best_latency - coarse_step_size
    end_latency = best_latency + coarse_step_size
    
    for latency in range(start_latency, end_latency + 1, fine_step_size):
        shifted_model = shift_audio(model, latency)
        mse = calculate_mse(amp, shifted_model)
        
        if mse < best_mse:
            best_mse = mse
            best_latency = latency
    
    return best_mse, best_latency

# Iterative volume adjustment
def find_min_lufs_volume(amp, model, rate, step_size=0.01, max_iters=1000):
    best_lufs = calculate_lufs(amp - model, rate)
    best_scale = 1.0
    
    for i in range(max_iters):
        scale_up = model * (1 + step_size)
        scale_down = model * (1 - step_size)
        
        lufs_up = calculate_lufs(amp - scale_up, rate)
        lufs_down = calculate_lufs(amp - scale_down, rate)
        
        if lufs_up < best_lufs:
            best_lufs = lufs_up
            model = scale_up
            best_scale *= (1 + step_size)
        elif lufs_down < best_lufs:
            best_lufs = lufs_down
            model = scale_down
            best_scale *= (1 - step_size)
        else:
            break    
    return best_lufs, best_scale

def find_min_lufs_combined(amp, model, rate, max_latency_samples=10000, volume_step_size=0.01, latency_step_size=1, max_volume_iters=1000):
    # Step 1: Adjust latency
    _, best_latency = calc_best_latency(amp, model, rate, max_latency_samples, latency_step_size*20)
    adjusted_model = shift_audio(model, best_latency)
    # Step 2: Adjust volume
    best_lufs_volume, best_scale = find_min_lufs_volume(amp, adjusted_model, rate, volume_step_size, max_volume_iters)
    
    return best_lufs_volume, best_scale, best_latency

def reduce_to_mono(audio):
    if len(audio.shape) == 2: 
        audio = np.mean(audio, axis=1)
    return audio

def peaq_score(original, reconstructed, sample_rate):
    if sample_rate != 16000: 
        original = signal.resample(original, int(len(original) * 16000 / sample_rate))
        reconstructed = signal.resample(reconstructed, int(len(reconstructed) * 16000 / sample_rate))
        sample_rate = 16000    
    pesq_mos = pesq(sample_rate, original, reconstructed, 'wb') 
    return pesq_mos

def signal_to_noise_ratio(original, reconstructed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def total_harmonic_distortion(original, reconstructed):
    original_harmonics = librosa.effects.harmonic(original)
    reconstructed_harmonics = librosa.effects.harmonic(reconstructed)
    residual = original_harmonics - reconstructed_harmonics
    thd = np.sum(residual ** 2) / np.sum(original_harmonics ** 2) * 100
    return thd

def spectral_flatness(original, reconstructed, sr, lowcut=20, highcut=20000):
    original = fft_bandpass_filter(original, sr, lowcut, highcut)
    reconstructed = fft_bandpass_filter(reconstructed, sr, lowcut, highcut)
    original_flatness = librosa.feature.spectral_flatness(y=original).mean()
    reconstructed_flatness = librosa.feature.spectral_flatness(y=reconstructed).mean()
    return original_flatness, reconstructed_flatness

def spectral_centroid_signal(signal, sr=48000, lowcut=20, highcut=20000):
    filtered_signal = fft_bandpass_filter(signal, sr, lowcut, highcut)
    return np.mean(librosa.feature.spectral_centroid(y=filtered_signal, sr=sr))

def spectral_centroid(original, reconstructed, sr, lowcut=20, highcut=20000):
    original_centroid = spectral_centroid_signal(original, sr=sr, lowcut=lowcut, highcut=highcut)
    reconstructed_centroid = spectral_centroid_signal(reconstructed, sr=sr, lowcut=lowcut, highcut=highcut)
    return original_centroid, reconstructed_centroid

def calculate_stats(inputfile, outputfile, difffile):
    amp_rate, amp_wav = wavfile.read(inputfile) 
    rate, model_wav = wavfile.read(outputfile)
     
    amp_wav = convert_to_float(amp_wav)
    model_wav = convert_to_float(model_wav)

    amp_wav = reduce_to_mono(amp_wav)
    model_wav = reduce_to_mono(model_wav)

    if amp_rate != rate:
        amp_wav = signal.resample(amp_wav, int(len(amp_wav) * rate / amp_rate))

    diff = len(amp_wav) - len(model_wav)
    if diff > 0:
        amp_wav = amp_wav[diff:]
    else:
        model_wav = model_wav[-diff:]

    amp = amp_wav.astype(np.float32)
    model = model_wav.astype(np.float32)
    best_lufs, best_scale, best_latency = find_min_lufs_combined(amp, model, rate, 5000)
    peaq = peaq_score(amp, model, rate)

    adjusted_model = model * best_scale
    adjusted_model = np.roll(adjusted_model, best_latency)

    if difffile != '':
        difference = amp[:len(adjusted_model)] - adjusted_model
        wavfile.write(difffile, rate, difference)

    mse = calculate_mse(amp, adjusted_model)
    snr = signal_to_noise_ratio(amp, adjusted_model)
    thd = total_harmonic_distortion(amp, adjusted_model)
    original_flatness, reconstructed_flatness = spectral_flatness(amp, adjusted_model, rate)
    original_centroid, reconstructed_centroid = spectral_centroid(amp, adjusted_model, rate)

    flatness_diff = reconstructed_flatness - original_flatness 
    centroid_diff = reconstructed_centroid - original_centroid

    combined = combined_score(best_lufs, best_scale, peaq, mse, snr, thd, flatness_diff, centroid_diff, original_flatness, original_centroid)

    results = {
        "best_lufs": best_lufs,
        "best_scale": best_scale,
        "best_latency": best_latency,
        "peaq": peaq,
        "mse": mse,
        "snr": snr,
        "thd": thd,
        "original_flatness": original_flatness,
        "reconstructed_flatness": reconstructed_flatness,
        "original_centroid": original_centroid,
        "reconstructed_centroid": reconstructed_centroid,
        "combined_score": combined
    }

    return results
