"""
Audio Signal Processing Script
Author: Kosta Tournavitis
Date: January 1, 2025

This script contains various functions for processing and analyzing audio signals. It includes methods for evaluating the quality of audio signals using metrics such as LUFS, SNR, THD, and PEAQ, as well as spectral analysis and filtering. Additionally, it provides functions for adjusting the latency and volume of audio signals.

Functions:
- combined_score: Calculates a combined quality score from multiple metrics.
- fft_bandpass_filter: Performs FFT bandpass filtering on an audio signal.
- adjust_latency: Adjusts the latency of a reconstructed signal to match the original.
- shift_audio: Shifts an audio signal by a certain number of samples.
- convert_to_float: Converts audio data to floating point.
- calculate_mse: Calculates the Mean Squared Error (MSE) between two audio signals.
- calculate_lufs: Calculates the loudness of an audio signal in LUFS.
- calc_best_latency: Finds the best latency to minimize MSE between two audio signals.
- find_min_lufs_volume: Finds the optimal volume to minimize the LUFS difference.
- find_min_lufs_combined: Combines latency and volume adjustments to minimize LUFS difference.
- reduce_to_mono: Reduces stereo signals to mono.
- peaq_score: Calculates the PEAQ score (Using PESQ) for two audio signals.
- signal_to_noise_ratio: Calculates the Signal-to-Noise Ratio (SNR) between two audio signals.
- total_harmonic_distortion: Calculates the Total Harmonic Distortion (THD) between two audio signals.
- spectral_flatness: Calculates the spectral flatness of original and reconstructed audio signals.
- spectral_centroid: Calculates the spectral centroid of original and reconstructed audio signals.
- calculate_stats: Computes and returns various metrics to evaluate audio signals.

This script is useful for analyzing and evaluating audio signals in various applications, including audio production, signal processing, and quality assurance.
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft, ifft
import pyloudnorm as pyln
from pesq import pesq
import librosa
import math

def combined_score(lufs, scale, peaq, mse, snr, thd, flatness_diff, centroid_diff, original_flatness, original_centroid):
    # Normalizing LUFS value
    lufs = max(lufs, -60)
    norm_lufs = 1.0-(lufs + 60) / 60
    norm_lufs = min(max(norm_lufs, 0.0), 1.0)

    # Normalizing other metrics
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
    
    # Defining weights for each metric
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

    # Calculating combined score using weighted sum of normalized metrics
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
    # Perform FFT on the signal
    fft_spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1/sr)
    
    # Zero out frequencies outside the desired band
    fft_spectrum[(frequencies < lowcut) | (frequencies > highcut)] = 0
    
    # Perform inverse FFT to get the filtered signal
    return np.real(np.fft.ifft(fft_spectrum))

def adjust_latency(original, reconstructed, latency):
    if latency > 0:
        # If latency is positive, pad the reconstructed signal at the beginning
        reconstructed = np.pad(reconstructed, (latency, 0), mode='constant')[:len(original)]
    elif latency < 0:
        # If latency is negative, pad the original signal at the beginning
        original = np.pad(original, (-latency, 0), mode='constant')[:len(reconstructed)]
    return original, reconstructed

def shift_audio(audio, samples):
    # Shift the audio signal by a certain number of samples
    if samples > 0:
        return np.pad(audio, (samples, 0), mode='constant')[:len(audio)]
    elif samples < 0:
        return np.pad(audio, (0, -samples), mode='constant')[abs(samples):]
    else:
        return audio

def convert_to_float(data):
    # Convert integer audio data to floating point
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

def calculate_mse(audio1, audio2):
    # Calculate Mean Squared Error between two audio signals
    return np.mean((audio1 - audio2) ** 2)

def calculate_lufs(audio, rate):
    # Calculate LUFS (Loudness Unit Full Scale) of the audio signal
    meter = pyln.Meter(rate)
    lufs  = meter.integrated_loudness(audio)
    return lufs

def calc_best_latency(amp, model, rate, max_latency_samples, coarse_step_size=10, fine_step_size=1):
    best_mse = calculate_mse(amp, model)
    best_latency = 0
    
    # Coarse search to find the best latency
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
    # Reduce stereo audio to mono by averaging the two channels
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    return audio

def peaq_score(original, reconstructed, sample_rate):
    if sample_rate != 16000: 
        # Resample audio if the sample rate is not 16000 Hz
        original = signal.resample(original, int(len(original) * 16000 / sample_rate))
        reconstructed = signal.resample(reconstructed, int(len(reconstructed) * 16000 / sample_rate))
        sample_rate = 16000
    
    # Calculate PEAQ score using PESQ
    pesq_mos = pesq(sample_rate, original, reconstructed, 'wb') 
    return pesq_mos

def signal_to_noise_ratio(original, reconstructed):
    # Calculate Signal-to-Noise Ratio (SNR) between original and reconstructed audio
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def total_harmonic_distortion(original, reconstructed):
    # Calculate Total Harmonic Distortion (THD) between original and reconstructed audio
    original_harmonics = librosa.effects.harmonic(original)
    reconstructed_harmonics = librosa.effects.harmonic(reconstructed)
    residual = original_harmonics - reconstructed_harmonics
    thd = np.sum(residual ** 2) / np.sum(original_h