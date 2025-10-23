import os
import numpy as np
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
import joblib

splits = ["train"]
vowels = ["A", "E", "I", "O", "U"]

mfcc_folder = "mfcc"
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

n_states = 5       # จำนวน state ของ HMM
n_mix = 3          # จำนวน Gaussian mixture components

for vowel in vowels:
    print(f"Training vowel '{vowel}'")
    X_concat = []
    lengths = []

    folder = os.path.join(mfcc_folder, "train", vowel)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}, skipping...")
        continue

    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue
        feat = np.load(os.path.join(folder, file))

        # normalize per utterance
        scaler = StandardScaler()
        feat = scaler.fit_transform(feat)

        X_concat.append(feat)
        lengths.append(feat.shape[0])

    if len(X_concat) == 0:
        print(f"No data for vowel {vowel}, skipping...")
        continue

    X_concat = np.vstack(X_concat)

    # train GMM-HMM
    model = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X_concat, lengths=lengths)

    out_path = os.path.join(model_folder, f"hmm_{vowel}.pkl")
    joblib.dump(model, out_path)
    print(f"Saved HMM model for vowel '{vowel}' -> {out_path}")
