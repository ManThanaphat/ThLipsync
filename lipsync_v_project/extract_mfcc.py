import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta

splits = ["train", "test"]
vowels = ["A", "E", "I", "O", "U"]

output_folder = "mfcc"
for split in splits:
    for vowel in vowels:
        os.makedirs(os.path.join(output_folder, split, vowel), exist_ok=True)

for split in splits:
    for vowel in vowels:
        folder = os.path.join(split, vowel)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}, skipping...")
            continue

        for file in os.listdir(folder):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(folder, file)
            rate, sig = wavfile.read(file_path)

            # MFCC
            mfcc_feat = mfcc(sig, rate, numcep=13, nfft=2048)
            # delta + delta-delta (ใช้ N=2)
            delta_feat = delta(mfcc_feat, N=2)
            delta2_feat = delta(delta_feat, N=2)
            feat = np.hstack([mfcc_feat, delta_feat, delta2_feat])

            out_path = os.path.join(output_folder, split, vowel, file.replace(".wav", ".npy"))
            np.save(out_path, feat)
