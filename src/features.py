import numpy as np
import neurokit2 as nk
from scipy.stats import entropy


def extract_features(window):

    features = {}

    # ---------------- HRV (ECG) ----------------
    try:
        ecg = window["ECG"].flatten()  

        ecg_clean = nk.ecg_clean(ecg, sampling_rate=700)
        peaks, _ = nk.ecg_peaks(ecg_clean, sampling_rate=700)
        hrv = nk.hrv(peaks, sampling_rate=700, show=False)

        features["SDNN"] = hrv["HRV_SDNN"].values[0]
        features["RMSSD"] = hrv["HRV_RMSSD"].values[0]

    except Exception as e:
        # print("HRV error:", e)  
        features["SDNN"] = 0
        features["RMSSD"] = 0

    # ---------------- EDA ----------------
    try:
        eda = window["EDA"].flatten()  

        eda_signals, eda_info = nk.eda_process(eda, sampling_rate=700)

        # Tonic component (baseline)
        features["SCL_mean"] = np.mean(eda_signals["EDA_Tonic"])

        # Phasic component (stress spikes)
        features["SCR_N"] = len(eda_info["SCR_Peaks"])

        # Raw variability
        features["EDA_std"] = np.std(eda)

    except Exception as e:
        # print("EDA error:", e)  
        features["SCL_mean"] = 0
        features["SCR_N"] = 0
        features["EDA_std"] = 0

    # ---------------- ACC ----------------
    try:
        acc = window["ACC"]  # (n,3)

        acc_mag = np.sqrt((acc ** 2).sum(axis=1))

        features["ACC_std"] = np.std(acc_mag)

        hist = np.histogram(acc_mag, bins=10)[0] + 1
        features["ACC_entropy"] = entropy(hist)

    except Exception as e:
        # print("ACC error:", e)
        features["ACC_std"] = 0
        features["ACC_entropy"] = 0

    # ---------------- Resp ----------------
    try:
        resp = window["Resp"].flatten() 

        features["Resp_mean"] = np.mean(resp)
        features["Resp_std"] = np.std(resp)

    except Exception as e:
        # print("Resp error:", e)
        features["Resp_mean"] = 0
        features["Resp_std"] = 0

    # ---------------- Temp ----------------
    try:
        temp = window["Temp"].flatten()  

        features["Temp_mean"] = np.mean(temp)

    except Exception as e:
        # print("Temp error:", e)
        features["Temp_mean"] = 0

    # ---------------- EMG ----------------
    try:
        emg = window["EMG"].flatten() 

        features["EMG_RMS"] = np.sqrt(np.mean(emg ** 2))

    except Exception as e:
        # print("EMG error:", e)
        features["EMG_RMS"] = 0

    # ---------------- Label ----------------
    features["Label"] = window["Label"]

    return features