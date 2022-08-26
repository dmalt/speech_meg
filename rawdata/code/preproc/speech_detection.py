from math import ceil

import numpy as np
from scipy.signal import medfilt  # type: ignore


def normalize(sound: np.ndarray) -> np.ndarray:
    max = sound.max()
    return sound / max if max > 0 else sound


def signal_energy(sound, window_length_samp):
    assert window_length_samp > 0, "Window length must be positive integer"
    sound = normalize(sound)

    num_frames = ceil(len(sound) / window_length_samp)
    result = np.empty(num_frames)
    lo, hi = 0, window_length_samp
    for i in range(num_frames):
        result[i] = np.square(sound[lo:hi]).mean()
        lo += window_length_samp
        hi += window_length_samp

    return result


def spectral_centroid(sound, window_length_samp):
    assert window_length_samp > 0, "Window length must be positive integer"
    num_frames = ceil(len(sound) / window_length_samp)
    result = np.empty(num_frames)
    sound = normalize(sound)

    lo, hi = 0, window_length_samp
    for i in range(num_frames):
        window = sound[lo:hi]
        window = np.multiply(window, np.hamming(len(window)))
        fft = np.fft.rfft(window)
        freq_sum = sum((j + 1) * np.abs(f) for j, f in enumerate(fft))
        fft_sum = np.abs(fft).sum()
        result[i] = freq_sum / fft_sum if fft_sum else 0
        print(f"{result[i]=}")

        lo += window_length_samp
        hi += window_length_samp

    return result


def smooth_signal(signal, window_length):
    result = medfilt(signal, window_length)
    result = medfilt(result, window_length)
    return result


def max_values(hist, num_values):
    max_args = [-1] * num_values
    bin_counts, bin_edges = hist[0], hist[1]
    bound = 0.02 * np.mean(bin_counts)

    i = 0
    while i < len(bin_counts):
        max_vals = [bin_counts[j] if j >= 0 else 0 for j in max_args]
        if (
            bin_counts[i] > np.min(max_vals)
            and (i == 0 or bin_counts[i] > bin_counts[i - 1])
            and (i == (len(bin_counts) - 1) or bin_counts[i] > bin_counts[i + 1])
            and bin_counts[i] > bound
        ):
            max_args[np.argmin(max_vals)] = i
            i += 2
        else:
            i += 1

    max_args = np.sort(max_args)  # type: ignore

    return np.array(
        [((bin_edges[i] + bin_edges[i + 1]) * 0.5) if i >= 0 else -1.0 for i in max_args]
    )


def calculate_threshold(signal, max, weight):
    if np.count_nonzero(max >= 0.0) < 2:
        return np.mean(signal)
    return (weight * max[0] + max[1]) / (weight + 1)


def post_process(mask, sound_length, window_length, extend_length):
    res = [0] * sound_length

    def set(start, end):
        start = start if start >= 0 else 0
        end = end if end <= sound_length else sound_length
        for i in range(start, end):
            res[i] = 1

    pos = 0
    for i in range(len(mask)):
        next_pos = pos + window_length
        if mask[i]:
            set(pos, next_pos)
        if i > 0 and mask[i] and not mask[i - 1]:
            set(pos - extend_length, pos)
        if i < (len(mask) - 1) and mask[i] and not mask[i + 1]:
            set(next_pos + 1, next_pos + extend_length)
        pos = next_pos

    return res


class SpeechDetector:
    def __init__(sound, sr):
        window_nsamp = round(0.025 * sr)
        num_bins = round(0.002 * sr)

        nrg = signal_energy(sound, window_nsamp)
        spc = spectral_centroid(sound, window_nsamp)

        nrgsm = smooth_signal(nrg, 5)
        nrghst = np.histogram(nrg, num_bins)
        maxnrg = max_values(nrghst, 2)
        nrgth = calculate_threshold(nrg, maxnrg, 5.0)

        spcsm = smooth_signal(spc, 5)
        spchst = np.histogram(spc, num_bins)
        maxspc = max_values(spchst, 2)
        spcth = calculate_threshold(spc, maxspc, 5.0)
        # print(f"{spcth=}")
        # spcth = 20
        nrgmsk = nrg > nrgth
        spcmsk = spc < spcth
        mask = post_process(
            np.logical_and(nrgmsk, spcmsk), len(sound), window_nsamp, 5 * window_nsamp
        )

    def _feature_mask(self, feature):
        nrgsm = smooth_signal(nrg, 5)
        nrghst = np.histogram(nrg, num_bins)
        maxnrg = max_values(nrghst, 2)
        nrgth = calculate_threshold(nrg, maxnrg, 5.0)

        # if draw:
        #     import matplotlib.pyplot as plt

        #     fig, axs = plt.subplots(2, 3)
        #     axs[0][0].plot(sound, linewidth=0.5)
        #     axs[1][0].plot(mask, linewidth=0.7, zorder=2)
        #     axs[1][0].plot(nrgmsk, linewidth=0.4, zorder=1)
        #     axs[1][0].plot(spcmsk, linewidth=0.4, zorder=1)
        #     axs[0][1].plot(nrg, linewidth=0.5, zorder=1)
        #     axs[0][1].plot(nrgsm, linewidth=0.5, zorder=2)
        #     axs[0][1].plot([0, len(nrg)], [nrgth, nrgth], linewidth=0.7, zorder=3)
        #     axs[1][1].plot(spc, linewidth=0.5, zorder=1)
        #     axs[1][1].plot(spcsm, linewidth=0.5, zorder=2)
        #     axs[1][1].plot([0, len(spc)], [spcth, spcth], linewidth=0.7, zorder=3)
        #     axs[0][2].bar(nrghst[1][:-1], nrghst[0], width=np.diff(nrghst[1]))
        #     axs[1][2].bar(spchst[1][:-1], spchst[0], width=np.diff(spchst[1]))
        #     plt.show()

        return mask


def remove_silence(sound, df):
    mask = detect_speech(sound, df)
    return np.multiply(sound, mask).astype(np.int16)


if __name__ == "__main__":
    import matplotlib
    import mne
    from mne import create_info
    from mne.io import RawArray
    from scipy.io.wavfile import read

    matplotlib.use("TkAgg")

    def annots_from_mask(mask: np.ndarray, sr: float, type: str) -> mne.Annotations:
        onset = []
        duration = []
        description = []
        prev_seg_start = 0 if mask[0] else None
        prev_sample = mask[0]
        for i, sample in enumerate(mask[1:], start=1):
            if not prev_sample and sample:
                prev_seg_start = i
            elif prev_sample and not sample:
                # one sample segment counts as zero-length
                assert prev_seg_start is not None
                onset.append(prev_seg_start / sr)
                duration.append((i - 1 - prev_seg_start) / sr)
                description.append(type)
                prev_seg_start = None
            prev_sample = sample
        if prev_seg_start is not None:
            onset.append(prev_seg_start / sr)
            duration.append((len(mask) - 1 - prev_seg_start) / sr)
            description.append(type)
        return mne.Annotations(onset, duration, description)

    sr, sound = read("sub-01_task-speech_proc-align_beh.wav")
    print(f"Read sound of length {len(sound) / sr} with sampling rate = {sr}")
    print(f"{sound=}")
    lo, hih = 30 * sr, 350 * sr
    sound = sound.astype(float)[lo:hih]
    print("Computing audio mask")
    mask = detect_speech(sound, sr, draw=True)
    print("Done")
    print(f"{len(mask)=}")

    info = create_info(ch_names=["audio"], sfreq=sr)
    raw = RawArray(sound[np.newaxis, :], info)
    annots = annots_from_mask(mask, sr=sr, type="speech")
    raw.set_annotations(annots)
    print("Downsampling ")
    raw.resample(sfreq=500)
    raw.plot(block=True)
