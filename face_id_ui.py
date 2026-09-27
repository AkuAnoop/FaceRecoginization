"""
Face ID Notch UI for macOS
Displays an iPhone-style Face ID overlay near the MacBook Pro notch.

Run:
    python3 face_id_ui.py enroll
    python3 face_id_ui.py auth
"""

import tkinter as tk
import threading
import time
import sys
import math

# ──────────────────────────────────────────────
# OVERLAY WINDOW
# ──────────────────────────────────────────────

class FaceIDOverlay:

    W, H       = 200, 230
    BRACKET    = 22    # bracket arm length
    BW         = 3     # bracket line width
    OVAL_W     = 90
    OVAL_H     = 110

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.0)   # start invisible
        self.root.configure(bg='black')

        screen_w = self.root.winfo_screenwidth()
        x = (screen_w // 2) - (self.W // 2)
        y = 4
        self.root.geometry(f"{self.W}x{self.H}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, width=self.W, height=self.H,
            bg='black', highlightthickness=0
        )
        self.canvas.pack()

        self.state      = 'idle'
        self.scan_y     = 0
        self.alpha      = 0.0
        self.pulse      = 0.0
        self.status_txt = "Looking for face…"
        self._running   = True

        self._tick()

    # ── public state setters ──────────────────

    def show_scanning(self):
        self.state      = 'scanning'
        self.status_txt = "Move closer…"
        self._fade_in()

    def show_authenticated(self):
        self.state      = 'success'
        self.status_txt = "Face ID"
        self.root.after(1800, self._fade_out)

    def show_failed(self):
        self.state      = 'failure'
        self.status_txt = "Not Recognised"
        self.root.after(1800, self._fade_out)

    def show_locked(self, seconds: int):
        self.state      = 'locked'
        self.status_txt = f"Try again in {seconds}s"
        self.root.after(1800, self._fade_out)

    def run(self):
        self.root.mainloop()

    def close(self):
        self._running = False
        self.root.destroy()

    # ── animation loop ────────────────────────

    def _tick(self):
        if not self._running:
            return
        self.scan_y  = (self.scan_y + 2.5) % (self.OVAL_H + 10)
        self.pulse   = (self.pulse + 0.07) % (2 * math.pi)
        self._draw()
        self.root.after(28, self._tick)

    def _fade_in(self):
        if self.alpha < 0.94:
            self.alpha = min(self.alpha + 0.07, 0.94)
            self.root.attributes('-alpha', self.alpha)
            self.root.after(20, self._fade_in)

    def _fade_out(self):
        if self.alpha > 0.0:
            self.alpha = max(self.alpha - 0.06, 0.0)
            self.root.attributes('-alpha', self.alpha)
            self.root.after(20, self._fade_out)
        else:
            self.state = 'idle'

    # ── drawing ───────────────────────────────

    def _draw(self):
        c  = self.canvas
        cx = self.W // 2
        cy = self.H // 2 - 10
        c.delete('all')

        # Background pill
        r = 18
        c.create_arc(6, 6, 6+r*2, 6+r*2, start=90, extent=90,  fill='#111111', outline='')
        c.create_arc(self.W-6-r*2, 6, self.W-6, 6+r*2, start=0, extent=90, fill='#111111', outline='')
        c.create_arc(6, self.H-6-r*2, 6+r*2, self.H-6, start=180, extent=90, fill='#111111', outline='')
        c.create_arc(self.W-6-r*2, self.H-6-r*2, self.W-6, self.H-6, start=270, extent=90, fill='#111111', outline='')
        c.create_rectangle(6+r, 6, self.W-6-r, self.H-6, fill='#111111', outline='')
        c.create_rectangle(6, 6+r, self.W-6, self.H-6-r, fill='#111111', outline='')

        # Color theme per state
        if self.state == 'success':
            color = '#30d158'    # Apple green
        elif self.state == 'failure' or self.state == 'locked':
            color = '#ff453a'    # Apple red
        else:
            pulse_bright = int(200 + 55 * math.sin(self.pulse))
            color = f'#{pulse_bright:02x}{pulse_bright:02x}{pulse_bright:02x}'

        ow, oh = self.OVAL_W // 2, self.OVAL_H // 2

        # Face oval
        c.create_oval(cx-ow, cy-oh, cx+ow, cy+oh, outline=color, width=1)

        # Scanning beam (only while scanning)
        if self.state == 'scanning':
            beam_y = cy - oh + self.scan_y
            if cy - oh <= beam_y <= cy + oh:
                c.create_rectangle(
                    cx-ow+2, beam_y-3, cx+ow-2, beam_y+3,
                    fill='#007aff', outline='', stipple='gray75'
                )
                c.create_rectangle(
                    cx-ow+2, beam_y-1, cx+ow-2, beam_y+1,
                    fill='#4dabff', outline=''
                )

        # Corner brackets
        bl = self.BRACKET
        bw = self.BW
        # top-left
        c.create_line(cx-ow,    cy-oh,    cx-ow+bl, cy-oh,    fill=color, width=bw, capstyle='round')
        c.create_line(cx-ow,    cy-oh,    cx-ow,    cy-oh+bl, fill=color, width=bw, capstyle='round')
        # top-right
        c.create_line(cx+ow,    cy-oh,    cx+ow-bl, cy-oh,    fill=color, width=bw, capstyle='round')
        c.create_line(cx+ow,    cy-oh,    cx+ow,    cy-oh+bl, fill=color, width=bw, capstyle='round')
        # bottom-left
        c.create_line(cx-ow,    cy+oh,    cx-ow+bl, cy+oh,    fill=color, width=bw, capstyle='round')
        c.create_line(cx-ow,    cy+oh,    cx-ow,    cy+oh-bl, fill=color, width=bw, capstyle='round')
        # bottom-right
        c.create_line(cx+ow,    cy+oh,    cx+ow-bl, cy+oh,    fill=color, width=bw, capstyle='round')
        c.create_line(cx+ow,    cy+oh,    cx+ow,    cy+oh-bl, fill=color, width=bw, capstyle='round')

        # Success checkmark
        if self.state == 'success':
            c.create_line(cx-18, cy+2,  cx-4,  cy+16, fill='#30d158', width=3, capstyle='round')
            c.create_line(cx-4,  cy+16, cx+18, cy-14, fill='#30d158', width=3, capstyle='round')

        # Failure X
        if self.state == 'failure' or self.state == 'locked':
            c.create_line(cx-14, cy-14, cx+14, cy+14, fill='#ff453a', width=3, capstyle='round')
            c.create_line(cx+14, cy-14, cx-14, cy+14, fill='#ff453a', width=3, capstyle='round')

        # Status label
        c.create_text(cx, self.H - 22, text=self.status_txt,
                      fill=color, font=('Helvetica Neue', 10), anchor='center')


# ──────────────────────────────────────────────
# FACE AUTH RUNNER  (background thread)
# ──────────────────────────────────────────────

def run_auth_in_background(overlay: FaceIDOverlay):
    """Runs face authentication on a background thread, updates overlay on main thread."""
    import cv2
    import numpy as np
    from face_id_recognition import FaceAuthenticator, FacePipeline, SecureStore
    from face_id_recognition import SIMILARITY_THRESHOLD, MAX_AUTH_ATTEMPTS, LOCKOUT_SECONDS

    store    = SecureStore()
    pipeline = FacePipeline()
    stored   = store.load()

    if stored is None:
        overlay.root.after(0, lambda: overlay.show_failed())
        return

    cap   = cv2.VideoCapture(0)
    start = time.time()
    matched = False
    failures = 0

    overlay.root.after(0, overlay.show_scanning)

    while time.time() - start < 15:
        ret, frame = cap.read()
        if not ret:
            continue

        embedding = pipeline.extract_embedding(frame)
        if embedding is not None:
            best = max(pipeline.cosine_similarity(embedding, s) for s in stored)
            if best >= SIMILARITY_THRESHOLD:
                matched = True
                break

    cap.release()

    if matched:
        overlay.root.after(0, overlay.show_authenticated)
        print("[AUTH] Authenticated.")
    else:
        overlay.root.after(0, overlay.show_failed)
        print("[AUTH] Not recognised.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "auth"

    if cmd == "enroll":
        from face_id_recognition import enroll
        enroll()

    elif cmd == "auth":
        overlay = FaceIDOverlay()
        t = threading.Thread(target=run_auth_in_background, args=(overlay,), daemon=True)
        t.start()
        overlay.run()

    else:
        print("Usage: python3 face_id_ui.py [enroll | auth]")
        sys.exit(1)
