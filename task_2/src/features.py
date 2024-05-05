import biosppy.signals.ecg as ecg
import neurokit2 as nk
import numpy as np
import pandas as pd
from biosppy.signals.tools import normalize


def feature_extraction(signal, sampling_freq):
    # Check if signal is inverted (and correct it).
    ecg_signal, _ = nk.ecg_invert(signal.dropna().values, sampling_rate=sampling_freq)

    # Filter ECG signal.
    cleaned_ecg = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_freq, method="biosppy"
    )
    cleaned_ecg = normalize(cleaned_ecg)[0]

    # Find peaks.
    rpeaks = ecg.engzee_segmenter(cleaned_ecg, sampling_rate=sampling_freq)["rpeaks"]
    _, waves_peak = nk.ecg_delineate(
        cleaned_ecg, rpeaks, sampling_rate=sampling_freq, method="peak"
    )

    feature_values = {}
    # Morphological features...
    # ... For R peaks.
    feature_values["R_mean"] = np.mean(cleaned_ecg[rpeaks])
    feature_values["R_med"] = np.median(cleaned_ecg[rpeaks])
    feature_values["R_sd"] = np.std(cleaned_ecg[rpeaks])

    # ... For the rest.
    waves = {}
    for wave_type in ["P", "Q", "S", "T"]:
        waves[wave_type] = np.array(waves_peak[f"ECG_{wave_type}_Peaks"])
        waves[wave_type] = waves[wave_type][~np.isnan(waves[wave_type])].astype(int)

        feature_values[f"{wave_type}_mean"] = np.mean(cleaned_ecg[waves[wave_type]])
        feature_values[f"{wave_type}_med"] = np.median(cleaned_ecg[waves[wave_type]])
        feature_values[f"{wave_type}_sd"] = np.std(cleaned_ecg[waves[wave_type]])

    # Time interval features.
    # R-R interval.
    RR_int = np.diff(rpeaks)
    feature_values["RR_int_mean"] = np.mean(RR_int)
    feature_values["RR_int_med"] = np.median(RR_int)
    feature_values["RR_int_sd"] = np.std(RR_int)

    # QRS interval.
    min_length = min(len(waves["Q"]), len(waves["S"]))
    QRS_int = waves["Q"][:min_length] - waves["S"][:min_length]
    feature_values["QRS_int_mean"] = np.mean(QRS_int)
    feature_values["QRS_int_med"] = np.median(QRS_int)
    feature_values["QRS_int_sd"] = np.std(QRS_int)

    # S-T interval.
    min_length = min(len(waves["T"]), len(waves["S"]))
    ST_int = waves["T"][:min_length] - waves["S"][:min_length]
    feature_values["ST_int_mean"] = np.mean(ST_int)
    feature_values["ST_int_med"] = np.median(ST_int)
    feature_values["ST_int_sd"] = np.std(ST_int)

    # P-Q interval.
    min_length = min(len(waves["P"]), len(waves["Q"]))
    PQ_int = waves["P"][:min_length] - waves["Q"][:min_length]
    feature_values["PQ_int_mean"] = np.mean(PQ_int)
    feature_values["PQ_int_med"] = np.median(PQ_int)
    feature_values["PQ_int_sd"] = np.std(PQ_int)

    # HRV features.
    hrv_time = (
        nk.hrv_time(rpeaks, sampling_rate=sampling_freq)
        .drop(
            columns=[
                "HRV_SDANN1",
                "HRV_SDNNI1",
                "HRV_SDANN2",
                "HRV_SDNNI2",
                "HRV_SDANN5",
                "HRV_SDNNI5",
            ]
        )
        .to_dict("records")[0]
    )
    hrv_freq = (
        nk.hrv_frequency(rpeaks, sampling_rate=sampling_freq)
        .drop(columns=["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_LFHF", "HRV_LFn"])
        .to_dict("records")[0]
    )

    feature_values.update(hrv_time)
    feature_values.update(hrv_freq)

    return pd.Series(feature_values)
