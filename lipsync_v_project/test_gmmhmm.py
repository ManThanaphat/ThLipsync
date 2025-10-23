import os
import numpy as np
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------- CONFIG ----------------
mfcc_folder = "mfcc"
model_folder = "models"
test_folder = os.path.join(mfcc_folder, "test")
results_summary_file = "test_results_summary.txt"
results_per_file_file = "test_results_per_file.txt"
vowels = ["A", "E", "I", "O", "U"]

# ---------------- LOAD MODELS ----------------
models = {}
for vowel in vowels:
    model_path = os.path.join(model_folder, f"hmm_{vowel}.pkl")
    if os.path.exists(model_path):
        models[vowel] = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model not found: {model_path}")

if not models:
    print("❌ No models found, exiting.")
    exit()

# ---------------- TEST LOOP ----------------
y_true = []
y_pred = []
file_results = []

for vowel in vowels:
    folder = os.path.join(test_folder, vowel)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}, skipping...")
        continue

    for file in os.listdir(folder):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(folder, file)
        feat = np.load(file_path)

        # Normalize per utterance
        scaler = StandardScaler()
        feat = scaler.fit_transform(feat)

        # Predict by max log-likelihood
        best_score = float("-inf")
        best_vowel = None
        scores_dict = {}
        for v, model in models.items():
            try:
                score = model.score(feat)
            except:
                score = float("-inf")
            scores_dict[v] = score
            if score > best_score:
                best_score = score
                best_vowel = v

        y_true.append(vowel)
        y_pred.append(best_vowel)

        # บันทึกผลรายไฟล์
        file_results.append({
            "file": file,
            "true": vowel,
            "pred": best_vowel,
            "scores": scores_dict
        })

# ---------------- SAVE SUMMARY ----------------
report = classification_report(y_true, y_pred, labels=vowels, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=vowels)
with open(results_summary_file, "w", encoding="utf-8") as f:
    f.write("รายงานการจำแนกสระ (Classification Report):\n")
    f.write(report)
    f.write("\nเมทริกซ์ความสับสน (Confusion Matrix):\n")
    f.write(np.array2string(cm))

print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)
print(f"\n✅ Saved summary to {results_summary_file}")

# ---------------- SAVE PER-FILE RESULTS ภาษาไทย ----------------
with open(results_per_file_file, "w", encoding="utf-8") as f:
    for r in file_results:
        scores_str = ", ".join([f"{k}:{v:.2f}" for k,v in r["scores"].items()])
        f.write(f"ไฟล์: {r['file']}\tสระจริง: {r['true']}\tสระทำนาย: {r['pred']}\tค่าความน่าจะเป็น (log-likelihood): [{scores_str}]\n")

print(f"✅ Saved per-file results (ภาษาไทย) to {results_per_file_file}")
