import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps, ImageFilter
import numpy as np
import soundfile as sf
import joblib
import librosa
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import pygame
import shutil
import subprocess
import tempfile
import os
import cv2
import math
import sys

# ---------------- Config ----------------

# ====== Robust PyInstaller-safe model loader + debug ======
import sys, traceback, logging
from pathlib import Path
import joblib
from tkinter import messagebox

# setup simple file logger (ไฟล์จะอยู่ข้าง exe ใน onedir หรือ temp ใน onefile)
try:
    if getattr(sys, "frozen", False):
        # บางครั้ง onefile ใช้ _MEIPASS, onedir จะไม่ต้องใช้ _MEIPASS
        base_try = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", None) else None
        if base_try and base_try.exists():
            BASE_DIR = base_try
        else:
            # fallback to folder where the executable lives (works for onedir)
            BASE_DIR = Path(sys.executable).resolve().parent
    else:
        BASE_DIR = Path(__file__).resolve().parent
except Exception:
    BASE_DIR = Path.cwd()

LOG_PATH = BASE_DIR / "lipsync_debug.log"
logging.basicConfig(filename=str(LOG_PATH), level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s: %(message)s")
logging.info("=== App start ===")
logging.info(f"Resolved BASE_DIR = {BASE_DIR}")

# set the folders
MODELS_DIR = BASE_DIR / "models"
MOUTH_SETS_DIR = BASE_DIR / "mouth_sets"
logging.info(f"MODELS_DIR = {MODELS_DIR}")
logging.info(f"MOUTH_SETS_DIR = {MOUTH_SETS_DIR}")

# find model files
models = {}
found_files = []
for ext in ("joblib","pkl"):
    for f in MODELS_DIR.glob(f"hmm_*.{ext}"):
        found_files.append(f)
        label = f.stem.split("_", 1)[1].upper() if "_" in f.stem else f.stem
        try:
            logging.info(f"Attempt loading {f}")
            models[label] = joblib.load(f)
            logging.info(f"✅ Loaded model: {f.name} as {label}")
        except Exception as e:
            logging.error(f"Failed to load {f}: {e}")
            logging.error(traceback.format_exc())

# debug prints + fallback message
if found_files:
    logging.info("Found model files:")
    for p in found_files:
        logging.info(f" - {p.name}")
else:
    logging.warning("No model files found in MODELS_DIR")

# If no models loaded, give user a clear message (useful in GUI/exe)
if not models:
    err_msg = (
        "No models loaded! ตรวจสอบว่าโฟลเดอร์ 'models' มีไฟล์ hmm_*.pkl หรือ hmm_*.joblib\n\n"
        f"Expected folder: {MODELS_DIR}\n\n"
        f"Debug log: {LOG_PATH}\n\n"
        "คำแนะนำ:\n"
        " - ถ้าใช้ PyInstaller แบบ --onefile ให้แน่ใจว่าใช้ --add-data \"models;models\" ด้วย\n"
        " - ถ้าใช้ --onedir ให้ตรวจดูว่าโฟลเดอร์ dist\\program\\models มีไฟล์จริง\n"
    )
    try:
        # ถ้าอยู่ใน GUI ให้โชว์ messagebox (จะทำงานทั้ง exe และ script)
        messagebox.showerror("Models not found", err_msg)
    except Exception:
        pass
    logging.error("No models loaded. Exiting.")
    raise FileNotFoundError("No models loaded! (ดู lipsync_debug.log)")

# ถ้าโหลดสำเร็จ ให้ logging ระบุ labels
logging.info(f"Loaded model labels: {list(models.keys())}")
# ====== end loader ======

MOUTH_SETS_DIR = BASE_DIR / "mouth_sets"
SR = 16000
N_MFCC = 13
N_FFT = 400
HOP_LENGTH = 160
FMIN = 50
FMAX = 7800
THRESH_DB = -30
MIN_SPEECH_LEN = 0.1
MOUTH_SETS_DIR = BASE_DIR / "mouth_sets"


# ---------------- Load models safely ----------------
models = {}

def load_models_safe(models_dir: Path):
    if not models_dir.exists():
        print(f"⚠️ โฟลเดอร์ models ไม่พบ: {models_dir}")
        return {}
    
    loaded = {}
    for ext in ["joblib", "pkl"]:
        for f in models_dir.glob(f"hmm_*.{ext}"):
            label = f.stem.split("_")[1].upper()
            try:
                print(f"กำลังโหลดโมเดล: {f}")
                loaded[label] = joblib.load(f)
                print(f"✅ โหลดสำเร็จ: {f.name}")
            except ModuleNotFoundError as me:
                print(f"❌ โหลดไม่สำเร็จ {f.name}: ไม่พบ module ที่ pickle ต้องการ: {me}")
            except Exception as e:
                print(f"❌ โหลดไม่สำเร็จ {f.name}: {e}")
    if not loaded:
        print(f"⚠️ ไม่มีโมเดลถูกโหลด! ตรวจสอบไฟล์ .pkl หรือ .joblib ในโฟลเดอร์ {models_dir}")
    return loaded

models = load_models_safe(MODELS_DIR)

if not models:
    # แทนที่จะ raise ทันที ให้ messagebox ช่วยแจ้งผู้ใช้
    from tkinter import Tk, messagebox
    root_temp = Tk()
    root_temp.withdraw()
    messagebox.showerror("Error", f"ไม่พบโมเดลที่โหลดได้! \nตรวจสอบไฟล์ .pkl หรือ .joblib ในโฟลเดอร์:\n{MODELS_DIR}")
    root_temp.destroy()
    sys.exit(1)

# ---------------- Audio / Feature utilities ----------------
def extract_mfcc39_from_signal(y, sr):
    if len(y) == 0:
        return np.zeros((0, N_MFCC*3), dtype=np.float32)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     fmin=FMIN, fmax=FMAX)
    n_frames = mfcc_feat.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames-1)
    if width < 3:
        d1 = np.zeros_like(mfcc_feat)
        d2 = np.zeros_like(mfcc_feat)
    else:
        d1 = librosa.feature.delta(mfcc_feat, order=1, width=width)
        d2 = librosa.feature.delta(mfcc_feat, order=2, width=width)
    feats = np.vstack([mfcc_feat, d1, d2]).T
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True)+1e-8)
    return feats.astype(np.float32)

def split_by_noise_gate(y, sr, threshold_db=THRESH_DB, min_speech_len=MIN_SPEECH_LEN):
    frame_len = int(0.025*sr)
    hop_len = int(0.010*sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    speech_mask = db > threshold_db
    times = librosa.frames_to_time(np.arange(len(speech_mask)), sr=sr, hop_length=hop_len)
    segments = []
    in_seg = False
    start_t = 0.0
    for t, active in zip(times, speech_mask):
        if active and not in_seg:
            start_t = t
            in_seg = True
        elif not active and in_seg:
            if t - start_t >= min_speech_len:
                segments.append((start_t, t))
            in_seg = False
    if in_seg:
        segments.append((start_t, times[-1]))
    return segments

def predict_segment(y_seg, sr):
    feats = extract_mfcc39_from_signal(y_seg, sr)
    scores = {}
    n_frames = max(1, feats.shape[0])
    for label, model in models.items():
        try:
            scores[label] = model.score(feats)/n_frames
        except:
            scores[label] = float("-inf")
    pred = max(scores, key=scores.get)
    return pred, scores

# ---------------- Mouth sets (image loader) ----------------
def load_mouth_sets():
    sets = {}
    if not MOUTH_SETS_DIR.exists():
        MOUTH_SETS_DIR.mkdir(parents=True, exist_ok=True)
    for folder in sorted(MOUTH_SETS_DIR.iterdir()):
        if folder.is_dir():
            imgs = {}
            for label in ["A","E","I","O","U","silent"]:
                f = folder / f"{label}.png"
                if f.exists():
                    try:
                        pil = Image.open(f)
                        # บีบอัดให้ fit 400x200 โดยไม่ยืดผิดสัดส่วน
                        pil.thumbnail((400,200), Image.LANCZOS)
                        imgs[label] = pil
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
            if imgs:
                sets[folder.name] = imgs
    return sets

# ---------------- UI color / style helper ----------------
PASTEL_BG = "#FFF7FB"       # very light pink cream
PANEL_BG = "#fff"           # card background white
ACCENT_1 = "#FFC9DE"        # pastel pinkdef load_mouth_sets():
ACCENT_2 = "#C7E7FF"        # pastel blue
ACCENT_3 = "#EAD6FF"        # pastel lavender
TEXT_MAIN = "#2b2b2b"
SUBTEXT = "#5a5a5a"
BUTTON_RADIUS = 8

def rounded_image(img_pil, radius=12, bg=(255,255,255)):
    """Return rounded-corner image for nicer UI rendering (PIL Image)."""
    w,h = img_pil.size
    mask = Image.new("L", (w,h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0,0,w,h), radius=radius, fill=255)
    out = Image.new("RGBA", (w,h), bg+(0,))
    out.paste(img_pil, (0,0))
    out.putalpha(mask)
    return out.convert("RGB")

# ---------------- GUI ----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg=PASTEL_BG)
        self.root.geometry("1120x640")
        self.root.minsize(880, 640)
        self.root.title("TH LipSync Studio")

        # ------- top header card -------
        header = tk.Frame(root, bg=PASTEL_BG)
        header.pack(fill="x", padx=18, pady=(14, 6))

        title_card = tk.Frame(header, bg=PANEL_BG, relief="flat", bd=0)
        title_card.pack(fill="x", padx=4, pady=4)
        title_card.configure(highlightthickness=0)

        title_frame = tk.Frame(title_card, bg=PANEL_BG)
        title_frame.pack(fill="x", padx=14, pady=12)

        lbl_title = tk.Label(title_frame, text="✨ Lipsync Studio", font=("Segoe UI", 20, "bold"), fg=TEXT_MAIN, bg=PANEL_BG)
        lbl_title.pack(side="left")

        # ------- main area -------
        main = tk.Frame(root, bg=PASTEL_BG)
        main.pack(fill="both", expand=True, padx=18, pady=(6,18))

        # left column: mouth preview + filename + play controls
        left = tk.Frame(main, bg=PASTEL_BG)
        left.pack(side="left", fill="y", padx=(0,12))

        self.label_filename = tk.Label(left, text="ยังไม่ได้เลือกไฟล์เสียง", font=("Arial", 12, "italic"), fg=SUBTEXT, bg=PASTEL_BG)
        self.label_filename.pack(pady=(2,8))

        # Mouth preview card
        preview_card = tk.Frame(left, bg=PANEL_BG, relief="flat", bd=0)
        preview_card.pack(pady=(0,12))
        preview_card.configure(padx=8, pady=8)

        self.label_img = tk.Label(preview_card, bg=PANEL_BG)
        self.label_img.pack()

        # control buttons
        ctrl_card = tk.Frame(left, bg=PASTEL_BG)
        ctrl_card.pack(pady=(8,6))

        btn_choose = tk.Button(ctrl_card, text="📂 เลือกไฟล์เสียง", command=self.open_file, bd=0,
                               font=("Segoe UI", 11), bg=ACCENT_2, fg=TEXT_MAIN, activebackground=ACCENT_2,
                               padx=10, pady=6)
        btn_choose.grid(row=0, column=0, padx=6, pady=6)

        btn_play = tk.Button(ctrl_card, text="▶️ เล่นเสียง + Animate", command=self.play, bd=0,
                             font=("Segoe UI", 11), bg=ACCENT_1, fg=TEXT_MAIN, activebackground=ACCENT_1,
                             padx=10, pady=6)
        btn_play.grid(row=0, column=1, padx=6, pady=6)

        btn_export = tk.Button(ctrl_card, text="🎬 ส่งออกวิดีโอ", command=self.export_video, bd=0,
                               font=("Segoe UI", 11), bg=ACCENT_3, fg=TEXT_MAIN, activebackground=ACCENT_3,
                               padx=10, pady=6)
        btn_export.grid(row=0, column=2, padx=6, pady=6)

        # right column: settings + mouth set + noise gate + results
        right = tk.Frame(main, bg=PASTEL_BG)
        right.pack(side="left", fill="both", expand=True)

        # mouth set selection card
        card_set = tk.Frame(right, bg=PANEL_BG)
        card_set.pack(fill="x", pady=(0,10), padx=4)
        card_set.configure(padx=12, pady=10)

        tk.Label(card_set, text="เลือกชุดปาก", font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="w")
        self.mouth_sets = load_mouth_sets()
        self.current_set = list(self.mouth_sets.keys())[0] if self.mouth_sets else None
        self.combo_set = ttk.Combobox(card_set, values=list(self.mouth_sets.keys()), state="readonly", width=24)
        self.combo_set.grid(row=0, column=1, padx=8)
        if self.current_set:
            self.combo_set.set(self.current_set)
        self.combo_set.bind("<<ComboboxSelected>>", self.change_set)

        btn_import_set = tk.Button(card_set, text="นำเข้าชุดใหม่", command=self.import_set, bd=0,
                                   font=("Segoe UI", 10), bg=ACCENT_2, fg=TEXT_MAIN, padx=8, pady=5)
        btn_import_set.grid(row=0, column=2, padx=6)

        # noise gate card
        card_noise = tk.Frame(right, bg=PANEL_BG)
        card_noise.pack(fill="x", pady=(0,10), padx=4)
        card_noise.configure(padx=12, pady=10)

        tk.Label(card_noise, text="Noise Gate (dB)", font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=TEXT_MAIN).grid(row=0, column=0, sticky="w")
        self.noise_var = tk.DoubleVar(value=THRESH_DB)
        # use Scale with command bound to update function (real-time)
        self.noise_scale = tk.Scale(card_noise, from_=-80, to=0, orient="horizontal",
                                    resolution=1, length=360, variable=self.noise_var,
                                    command=self.update_noisegate, bg=PANEL_BG, troughcolor=ACCENT_3)
        self.noise_scale.grid(row=0, column=1, padx=8, pady=6, sticky="w")

        # show current numeric value
        self.noise_value_label = tk.Label(card_noise, text=f"{THRESH_DB} dB", font=("Segoe UI", 10), bg=PANEL_BG, fg=SUBTEXT)
        self.noise_value_label.grid(row=0, column=2, padx=6)

        # result card (seg list)
        card_result = tk.Frame(right, bg=PANEL_BG)
        card_result.pack(fill="both", expand=True, pady=(0,6), padx=4)
        card_result.configure(padx=10, pady=10)

        tk.Label(card_result, text="ผลการทำนาย (Sequence)", font=("Segoe UI", 11, "bold"), bg=PANEL_BG, fg=TEXT_MAIN).pack(anchor="w")
        self.result_text = tk.Text(card_result, height=12, wrap="word", font=("Consolas", 11), bg="#FFF", fg=TEXT_MAIN)
        self.result_text.pack(fill="both", expand=True, pady=(6,0))

        # footer help
        footer = tk.Frame(root, bg=PASTEL_BG)
        footer.pack(fill="x", padx=18, pady=(6,12))
        tk.Label(footer, text="Tip: เลื่อน Noise Gate เพื่ออัปเดตการแยกเสียงและการทำนายแบบเรียลไทม์", font=("Segoe UI", 9), bg=PASTEL_BG, fg=SUBTEXT).pack(side="left")

        # init preview image
        img_close = self.draw_mouth(None)
        self.label_img.config(image=img_close)
        self.label_img.image = img_close

        # internal state
        self.audio_file_path = None
        self.audio_data = None
        self.sr = SR
        self.segments = []
        self.seg_preds = []
        self.seg_times = []
        pygame.mixer.init()

    # ---------------- Update on Noise Gate change ----------------
    def update_noisegate(self, val):
        # val is a string from Scale
        if self.audio_data is None:
            # update numeric label only
            try:
                self.noise_value_label.config(text=f"{float(val):.0f} dB")
            except:
                pass
            return

        try:
            threshold = float(val)
        except:
            threshold = THRESH_DB
        # update numeric label
        self.noise_value_label.config(text=f"{threshold:.0f} dB")

        seg_intervals = split_by_noise_gate(self.audio_data, SR, threshold, MIN_SPEECH_LEN)

        self.segments, self.seg_preds, self.seg_times = [], [], []
        for start, end in seg_intervals:
            s = int(round(start * SR))
            e = int(round(end * SR))
            seg_y = self.audio_data[s:e]
            if len(seg_y) == 0:
                continue
            pred, _ = predict_segment(seg_y, SR)
            self.segments.append(seg_y)
            self.seg_preds.append(pred)
            self.seg_times.append((start, end))

        # show results succinctly
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Noise Gate: {threshold:.0f} dB\n")
        if not self.seg_preds:
            self.result_text.insert(tk.END, "ไม่มีการตรวจจับเสียง (ค่า threshold อาจสูงเกินไป)\n")
        else:
            seq = "  ".join(self.seg_preds)
            self.result_text.insert(tk.END, f"Pred sequence ({len(self.seg_preds)} segments):\n{seq}\n\n")
            for i, ((st,en), p) in enumerate(zip(self.seg_times, self.seg_preds), start=1):
                self.result_text.insert(tk.END, f"{i}. {p}  —  {st:.2f}s to {en:.2f}s\n")

    # ---------------- Audio loading ----------------
    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files","*.wav"), ("All files","*.*")])
        if not path:
            return
        self.audio_file_path = path
        self.label_filename.config(text=f"🎵 {Path(path).name}")
        y, sr0 = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr0 != SR:
            y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
        self.audio_data = y
        # run initial segmentation with current slider value
        self.update_noisegate(self.noise_var.get())

    # ---------------- Play + animate ----------------
    def play(self):
        if not self.audio_file_path:
            messagebox.showinfo("ไม่มีไฟล์", "กรุณาเลือกไฟล์เสียงก่อนเล่น")
            return
        try:
            sound = pygame.mixer.Sound(self.audio_file_path)
            sound.play()
        except Exception as e:
            messagebox.showwarning("เล่นเสียงไม่สำเร็จ", f"ไม่สามารถเล่นไฟล์เสียงได้: {e}")
            return

        self.label_filename.config(text=f"🎧 เล่น: {Path(self.audio_file_path).name}")
        # animate in sequence (uses predicted segments)
        for pred, seg_y in zip(self.seg_preds, self.segments):
            mouth_img = self.draw_mouth(pred)
            self.label_img.config(image=mouth_img)
            self.label_img.image = mouth_img
            self.root.update()
            dur_ms = int(len(seg_y) / SR * 1000)
            pygame.time.wait(max(20, dur_ms))  # ensure not zero

        img_close = self.draw_mouth(None)
        self.label_img.config(image=img_close)
        self.label_img.image = img_close
        self.label_filename.config(text=f"✅ เล่นจบ: {Path(self.audio_file_path).name}")

    # ---------------- Export video ----------------
    def export_video(self):
        if not self.audio_file_path:
            messagebox.showwarning("ยังไม่มีไฟล์", "กรุณาเลือกไฟล์เสียงก่อนส่งออก")
            return
        if not self.seg_times or not self.seg_preds:
            messagebox.showwarning("ยังไม่มีข้อมูล", "กรุณากด 'เลือกไฟล์เสียง' เพื่อวิเคราะห์ก่อนส่งออก")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                 filetypes=[("MP4 video", "*.mp4")],
                                                 title="บันทึกวิดีโอเป็น...")
        if not save_path:
            return

        # read full audio to ensure alignment
        y_full, sr0 = sf.read(self.audio_file_path)
        if y_full.ndim > 1:
            y_full = y_full.mean(axis=1)
        if sr0 != SR:
            y_full = librosa.resample(y_full, orig_sr=sr0, target_sr=SR)
        duration = len(y_full) / SR

        frame_rate = 25
        total_frames = int(math.ceil(duration * frame_rate))

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir)
            for i in range(total_frames):
                t_now = i / frame_rate
                current_pred = "silent"
                for (start, end), pred in zip(self.seg_times, self.seg_preds):
                    if start <= t_now < end:
                        current_pred = pred
                        break
                pil_img = self.get_mouth_image(current_pred)
                pil_img_rgb = pil_img.convert("RGB")
                frame = np.array(pil_img_rgb)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_file = frame_dir / f"frame_{i:06d}.png"
                cv2.imwrite(str(frame_file), frame)

            video_tmp = frame_dir / "temp_video.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(frame_rate),
                "-i", str(frame_dir / "frame_%06d.png"),
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(video_tmp)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_tmp),
                "-i", self.audio_file_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(save_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        messagebox.showinfo("สำเร็จ", f"ส่งออกวิดีโอพร้อมเสียงเรียบร้อย!\n{save_path}")

    # ---------------- Mouth (image) helpers ----------------
    def draw_mouth(self, label):
        L = label.upper() if label else "silent"
        imgs = self.mouth_sets.get(self.current_set, {})

        if L in imgs:
            pil = imgs[L]
        elif "silent" in imgs:
            pil = imgs["silent"]
        else:
            # fallback simple rectangle
            fig, ax = plt.subplots(figsize=(4,2))
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
            mouth = Rectangle((0.35,0.48), 0.3, 0.02)
            mouth.set_facecolor((0.9,0.2,0.2))
            ax.add_patch(mouth)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
            plt.close(fig)
            buf.seek(0)
            pil = Image.open(buf)
            pil.thumbnail((400,200), Image.LANCZOS)

        # ทำ rounded corner
        pil_rounded = rounded_image(pil, radius=14)
        return ImageTk.PhotoImage(pil_rounded)


    def change_set(self, event=None):
        self.current_set = self.combo_set.get()
        img = self.draw_mouth(None)
        self.label_img.config(image=img)
        self.label_img.image = img

    def import_set(self):
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ชุดปาก")
        if not folder:
            return
        name = Path(folder).name
        dest = MOUTH_SETS_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        for label in ["A","E","I","O","U","silent"]:
            src = Path(folder) / f"{label}.png"
            if src.exists():
                shutil.copy(src, dest / f"{label}.png")
        self.mouth_sets = load_mouth_sets()
        self.combo_set["values"] = list(self.mouth_sets.keys())
        self.combo_set.set(name)
        self.current_set = name
        messagebox.showinfo("สำเร็จ", f"นำเข้าชุดใหม่ '{name}' แล้วค่ะ 💕")

    def get_mouth_image(self, pred):
        # return PIL.Image
        if not self.current_set or self.current_set not in self.mouth_sets:
            return Image.new("RGB", (400,200), (240,240,240))
        imgs = self.mouth_sets[self.current_set]
        label = pred if pred in imgs else "silent"
        img = imgs.get(label, imgs.get("silent"))
        return img

# ---------------- Run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
