import numpy as np
import neurokit2 as nk
from scipy.stats import entropy

def extract_features(window):
    
    features = {}
    
    # ---------------- HRV (ECG) ----------------
    try:
        ecg = window["ECG"]
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=700)
        peaks, _ = nk.ecg_peaks(ecg_clean, sampling_rate=700)
        hrv = nk.hrv(peaks, sampling_rate=700, show=False)
        
        features["SDNN"] = hrv["HRV_SDNN"].values[0]
        features["RMSSD"] = hrv["HRV_RMSSD"].values[0]
    except:
        features["SDNN"] = 0
        features["RMSSD"] = 0

    # ---------------- EDA ----------------
    try:
        eda = window["EDA"]
        
        eda_signals, eda_info = nk.eda_process(eda, sampling_rate=700)
        
        # Tonic component (baseline)
        features["SCL_mean"] = np.mean(eda_signals["EDA_Tonic"])
        
        # Phasic component (stress spikes)
        features["SCR_N"] = len(eda_info["SCR_Peaks"])
        
        features["EDA_std"] = np.std(eda)
        
    except:
        features["SCL_mean"] = 0
        features["SCR_N"] = 0
        features["EDA_std"] = 0

    # ---------------- ACC ----------------
    acc = window["ACC"]
    acc_mag = np.sqrt((acc**2).sum(axis=1))
    
    features["ACC_std"] = np.std(acc_mag)
    features["ACC_entropy"] = entropy(np.histogram(acc_mag, bins=10)[0] + 1)

    # ---------------- Resp ----------------
    resp = window["Resp"]
    features["Resp_mean"] = np.mean(resp)
    features["Resp_std"] = np.std(resp)

    # ---------------- Temp ----------------
    temp = window["Temp"]
    features["Temp_mean"] = np.mean(temp)

    # ---------------- EMG ----------------
    emg = window["EMG"]
    features["EMG_RMS"] = np.sqrt(np.mean(emg**2))

    # ---------------- Label ----------------
    features["Label"] = window["Label"]

    return features