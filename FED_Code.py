import cv2
from deepface import DeepFace
import threading
import time
import sys
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import os
import datetime
import pyttsx3
import numpy as np 

# ------------------------------
# Configuration (change if you want)
# ------------------------------
NEW_WIDTH = 1280
NEW_HEIGHT = 720
CAMERA_IDS = [0, 1, 2]  # cycle through these IDs
DEFAULT_CAMERA_INDEX = 0
DETECTOR_BACKENDS = ['opencv', 'ssd', 'retinaface']
DEFAULT_BACKEND = 'opencv'
SCREENSHOT_DIR = "screenshots"
SPEECH_COOLDOWN = 1.5  # seconds between speaking same emotion

# ensure screenshot folder exists
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------
# Globals / State
# ------------------------------
window = tk.Tk()
window.title("Real-time Face Emotion Detector")
window.geometry("1400x820")
is_running = False
camera = None
camera_index = DEFAULT_CAMERA_INDEX
detector_backend = DEFAULT_BACKEND
fps = 0.0
last_spoken_emotion = None
last_spoken_time = 0.0
theme = "dark"
use_fullscreen = False

# initialize speech engine
try:
    speech_engine = pyttsx3.init()
    speech_engine.setProperty('rate', 150)
except Exception as e:
    speech_engine = None
    print("Warning: pyttsx3 speech engine not available:", e)

# ------------------------------
# UI Layout
# ------------------------------
# top frame: controls
top_frame = tk.Frame(window, bg="#1e1e1e")
top_frame.pack(side="top", fill="x", padx=6, pady=6)

video_frame = tk.Frame(window, bg="black", width=960, height=680)
video_frame.pack(side="left", padx=8, pady=8)
video_frame.pack_propagate(False)

video_label = tk.Label(video_frame, bg="black")
video_label.pack(expand=True)

# right panel for multi-face cards & status
right_panel = tk.Frame(window, bg="#2b2b2b", width=360)
right_panel.pack(side="right", fill="y", padx=8, pady=8)
right_panel.pack_propagate(False)

cards_frame = tk.Frame(right_panel, bg=right_panel['bg'])
cards_frame.pack(fill="both", expand=True, padx=6, pady=6)

status_label = tk.Label(right_panel, text="Status: Stopped", bg=right_panel['bg'], fg="white", anchor="w")
status_label.pack(side="bottom", fill="x", padx=6, pady=6)

# ------------------------------
# Controls (top)
# ------------------------------
def create_button(parent, text, cmd, bg="#4caf50"):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg="white", font=("Arial", 11), padx=8, pady=6)

start_btn = create_button(top_frame, "Start Camera", lambda: start_camera(), bg="#2e7d32")
start_btn.pack(side="left", padx=6)

stop_btn = create_button(top_frame, "Stop Camera", lambda: stop_camera(), bg="#b71c1c")
stop_btn.pack(side="left", padx=6)

screenshot_btn = create_button(top_frame, "Save Screenshot", lambda: save_screenshot(), bg="#1565c0")
screenshot_btn.pack(side="left", padx=6)

switch_cam_btn = create_button(top_frame, "Switch Camera", lambda: switch_camera(), bg="#6a1b9a")
switch_cam_btn.pack(side="left", padx=6)

# backend dropdown
backend_label = tk.Label(top_frame, text="Backend:", bg=top_frame['bg'], fg="white", font=("Arial", 10))
backend_label.pack(side="left", padx=(20, 4))
backend_var = tk.StringVar(value=detector_backend)
backend_menu = ttk.OptionMenu(top_frame, backend_var, detector_backend, *DETECTOR_BACKENDS,
                              command=lambda v: set_backend(v))
backend_menu.pack(side="left")

# theme toggle
theme_btn = create_button(top_frame, "Toggle Theme", lambda: toggle_theme(), bg="#455a64")
theme_btn.pack(side="left", padx=8)

# fullscreen toggle
fs_btn = create_button(top_frame, "Toggle Fullscreen", lambda: toggle_fullscreen(), bg="#00796b")
fs_btn.pack(side="left", padx=6)

# speech toggle
speech_var = tk.BooleanVar(value=True)
speech_check = tk.Checkbutton(top_frame, text="Speech", variable=speech_var, bg=top_frame['bg'], fg="white")
speech_check.pack(side="left", padx=8)

# ------------------------------
# Theme functions
# ------------------------------
def apply_theme():
    global theme
    if theme == "dark":
        window.configure(bg="#1e1e1e")
        top_frame.configure(bg="#1e1e1e")
        video_frame.configure(bg="black")
        right_panel.configure(bg="#2b2b2b")
        status_label.configure(bg=right_panel['bg'], fg="white")
        for widget in top_frame.winfo_children():
            if isinstance(widget, tk.Label) or isinstance(widget, tk.Checkbutton):
                widget.configure(bg=top_frame['bg'], fg="white")
    else:
        window.configure(bg="#f3f3f3")
        top_frame.configure(bg="#f3f3f3")
        video_frame.configure(bg="#ddd")
        right_panel.configure(bg="#e9e9e9")
        status_label.configure(bg=right_panel['bg'], fg="black")
        for widget in top_frame.winfo_children():
            if isinstance(widget, tk.Label) or isinstance(widget, tk.Checkbutton):
                widget.configure(bg=top_frame['bg'], fg="black")

def toggle_theme():
    global theme
    theme = "light" if theme == "dark" else "dark"
    apply_theme()

apply_theme()

# ------------------------------
# Camera / Control Functions
# ------------------------------
def set_backend(value):
    global detector_backend
    detector_backend = value
    status_label.config(text=f"Status: Backend set to {detector_backend}")

def switch_camera():
    global camera_index, camera, is_running
    # cycle through CAMERA_IDS
    try:
        idx = CAMERA_IDS.index(camera_index)
        next_idx = (idx + 1) % len(CAMERA_IDS)
    except ValueError:
        next_idx = 0
    camera_index = CAMERA_IDS[next_idx]
    status_label.config(text=f"Status: Camera switched to {camera_index}")
    if is_running:
        # restart camera with new index
        stop_camera()
        time.sleep(0.2)
        start_camera()

def start_camera():
    global is_running, camera, fps, camera_index
    if is_running:
        return
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        status_label.config(text=f"Status: Could not open camera {camera_index}")
        return
    is_running = True
    status_label.config(text=f"Status: Running (camera {camera_index}, backend {detector_backend})")
    threading.Thread(target=process_frames, daemon=True).start()

def stop_camera():
    global is_running, camera
    is_running = False
    if camera:
        try:
            camera.release()
        except Exception:
            pass
    video_label.config(image="")
    clear_cards()
    status_label.config(text="Status: Stopped")

# ------------------------------
# Screenshot
# ------------------------------
def save_screenshot():
    # grabs current frame from label image (if available) and saves PNG
    # fallback: request a fresh frame capture if none
    try:
        img = video_label.imgtk  # this is PhotoImage object
        # convert PhotoImage back to PIL via image attribute if available
        # but PhotoImage doesn't expose image bytes easily; instead, store last_frame globally in loop
        if 'last_frame_image' in globals() and last_frame_image is not None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
            last_frame_image.save(path)
            status_label.config(text=f"Status: Screenshot saved -> {path}")
        else:
            status_label.config(text="Status: No frame available to save")
    except Exception as e:
        status_label.config(text=f"Status: Screenshot failed: {e}")

# ------------------------------
# Multi-face cards helpers
# ------------------------------
card_widgets = []

def clear_cards():
    global card_widgets
    for w in card_widgets:
        w.destroy()
    card_widgets = []

def update_cards(faces):
    """
    faces: list of dicts with keys: 'img' (PIL.Image), 'emotion', 'prob'
    """
    global card_widgets
    # remove older widgets
    clear_cards()
    if not faces:
        lbl = tk.Label(cards_frame, text="No faces detected", bg=cards_frame['bg'], fg="white")
        lbl.pack(pady=6)
        card_widgets.append(lbl)
        return

    for i, f in enumerate(faces):
        frame = tk.Frame(cards_frame, bg=cards_frame['bg'], bd=1, relief="solid")
        frame.pack(fill="x", padx=4, pady=6)

        # thumbnail (resize to small)
        thumb = f['img'].copy()
        thumb.thumbnail((96, 96))
        thumb_tk = ImageTk.PhotoImage(thumb)
        thumb_lbl = tk.Label(frame, image=thumb_tk, bg=frame['bg'])
        # keep ref
        thumb_lbl.image = thumb_tk
        thumb_lbl.pack(side="left", padx=6, pady=6)

        text_frame = tk.Frame(frame, bg=frame['bg'])
        text_frame.pack(side="left", expand=True, fill="y", padx=6)
        lbl_em = tk.Label(text_frame, text=f"Emotion: {f['emotion']}", anchor="w", bg=frame['bg'], fg="white", font=("Arial", 11, "bold"))
        lbl_em.pack(anchor="w")
        lbl_pr = tk.Label(text_frame, text=f"Prob: {f['prob']:.2f}%", anchor="w", bg=frame['bg'], fg="lightgray", font=("Arial", 10))
        lbl_pr.pack(anchor="w")
        card_widgets.append(frame)

# ------------------------------
# Speech (non-blocking)
# ------------------------------
def speak_emotion(emotion_text):
    if not speech_engine:
        return
    def _s():
        try:
            speech_engine.say(emotion_text)
            speech_engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)
    threading.Thread(target=_s, daemon=True).start()

# ------------------------------
# Main processing loop
# ------------------------------
def process_frames():
    global is_running, fps, last_spoken_emotion, last_spoken_time, last_frame_image
    last_frame_image = None
    last_time = time.time()

    while is_running:
        loop_start = time.time()
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Resize for consistent processing/display
        if NEW_WIDTH > 0 and NEW_HEIGHT > 0:
            frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))

        faces_for_cards = []
        dominant_to_speak = None
        try:
            # analyze - allow both single and multiple faces
            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=detector_backend
            )

            # normalize results to list
            if isinstance(results, dict):
                results = [results]

            # For each face, draw box and label
            for res in results:
                # ensure region keys exist and are valid ints
                region = res.get('region', {})
                x = int(region.get('x', 0))
                y = int(region.get('y', 0))
                w = int(region.get('w', 0))
                h = int(region.get('h', 0))
                emo = res.get('dominant_emotion', 'unknown')
                # sometimes emotion dict keys are words -> get probability
                emotion_dict = res.get('emotion', {})
                prob = emotion_dict.get(emo, 0.0) if isinstance(emotion_dict, dict) else 0.0

                # bounding box clamp
                x = max(0, x); y = max(0, y); w = max(0, w); h = max(0, h)

                # draw rectangle and labels
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emo, (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"{prob:.2f}%", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # prepare face thumbnail for cards (crop safely)
                try:
                    crop = frame[y:y + h, x:x + w].copy()
                    if crop.size == 0:
                        # fallback: create small blank
                        crop = 255 * (np.ones((96,96,3), dtype='uint8'))
                    # convert to PIL image (RGB)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)
                except Exception:
                    # fallback create small blank image
                    pil_img = Image.new("RGB", (96, 96), color=(100, 100, 100))

                faces_for_cards.append({'img': pil_img, 'emotion': emo, 'prob': prob})

            # pick primary face to possibly speak (first detected)
            if faces_for_cards:
                dominant_to_speak = faces_for_cards[0]['emotion']

        except Exception as e:
            # DeepFace can raise many exceptions; ignore but optionally log
            # print("Analyze error:", e)
            pass

        # FPS smoothing
        loop_end = time.time()
        loop_time = loop_end - loop_start if loop_end - loop_start > 0 else 1e-6
        curr_fps = 1.0 / loop_time
        fps = 0.9 * fps + 0.1 * curr_fps if fps > 0 else curr_fps

        cv2.putText(frame, f"FPS: {fps:.1f} ({detector_backend})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # prepare PIL image to display and to save if needed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        last_frame_image = pil_img  # keep for screenshot

        imgtk = ImageTk.PhotoImage(image=pil_img.resize((960, int(960 * pil_img.height / pil_img.width))))
        # update GUI in thread-safe manner via after
        def _update_gui(img=imgtk):
            try:
                video_label.imgtk = img
                video_label.configure(image=img)
            except Exception:
                pass
        window.after(0, _update_gui)

        # update cards panel (do minimal UI work)
        try:
            update_cards(faces_for_cards)
        except Exception:
            pass

        # speak logic: only speak when changed and cooldown passed
        now = time.time()
        if speech_var.get() and dominant_to_speak:
            if dominant_to_speak != last_spoken_emotion or (now - last_spoken_time) > SPEECH_COOLDOWN:
                # speak on change
                last_spoken_emotion = dominant_to_speak
                last_spoken_time = now
                if speech_engine:
                    speak_emotion(f"{dominant_to_speak}")
        # small sleep to yield CPU
        # but keep low latency
        time.sleep(0.01)

    # end of while is_running
    print("Frame loop stopped")

# ------------------------------
# Window Close Handler
# ------------------------------
def on_close():
    stop_camera()
    try:
        if speech_engine:
            speech_engine.stop()
    except Exception:
        pass
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_close)

# ------------------------------
# Fullscreen toggle
# ------------------------------
def toggle_fullscreen():
    global use_fullscreen
    use_fullscreen = not use_fullscreen
    if use_fullscreen:
        # borderless fullscreen
        window.attributes("-fullscreen", True)
        window.overrideredirect(True)
    else:
        window.attributes("-fullscreen", False)
        window.overrideredirect(False)

# ------------------------------
# Start UI
# ------------------------------
if __name__ == "__main__":
    window.mainloop()