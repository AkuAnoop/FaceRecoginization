"""
Face ID Recognition System for macOS
Mirrors the enrollment → liveness → embedding → match pipeline used by Apple Face ID.

Dependencies:
    pip install opencv-python face_recognition numpy keyring cryptography

Hardware: FaceTime HD camera (RGB only — no TrueDepth)
Storage:  System Keychain via `keyring` (wraps macOS Keychain)
"""

import cv2
import face_recognition
import numpy as np
import keyring
import json
import time
import sys
from cryptography.fernet import Fernet
import base64
import hashlib

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

SERVICE_NAME        = "MacFaceID"
KEYCHAIN_KEY        = "face_embeddings"
ENCRYPTION_KEY_ID   = "face_enc_key"

SIMILARITY_THRESHOLD    = 0.70   # cosine similarity — raise for stricter (max 1.0)
ENROLLMENT_FRAMES       = 10     # frames captured during setup
MAX_AUTH_ATTEMPTS       = 5      # lockout after this many failures
LOCKOUT_SECONDS         = 30     # seconds locked out after max failures
LIVENESS_BLINK_REQUIRED = True   # require at least one blink during auth


# ──────────────────────────────────────────────
# SECURE STORAGE  (macOS Keychain via keyring)
# ──────────────────────────────────────────────

class SecureStore:
    """Encrypts embeddings and stores them in the macOS Keychain."""

    def _get_or_create_key(self) -> bytes:
        raw = keyring.get_password(SERVICE_NAME, ENCRYPTION_KEY_ID)
        if raw is None:
            key = Fernet.generate_key()
            keyring.set_password(SERVICE_NAME, ENCRYPTION_KEY_ID, key.decode())
            return key
        return raw.encode()

    def save(self, embeddings: list[np.ndarray]) -> None:
        key  = self._get_or_create_key()
        f    = Fernet(key)
        data = json.dumps([e.tolist() for e in embeddings]).encode()
        encrypted = f.encrypt(data)
        keyring.set_password(SERVICE_NAME, KEYCHAIN_KEY, encrypted.decode())

    def load(self) -> list[np.ndarray] | None:
        raw = keyring.get_password(SERVICE_NAME, KEYCHAIN_KEY)
        if raw is None:
            return None
        key  = self._get_or_create_key()
        f    = Fernet(key)
        data = f.decrypt(raw.encode())
        return [np.array(e) for e in json.loads(data)]

    def clear(self) -> None:
        keyring.delete_password(SERVICE_NAME, KEYCHAIN_KEY)
        keyring.delete_password(SERVICE_NAME, ENCRYPTION_KEY_ID)


# ──────────────────────────────────────────────
# LIVENESS DETECTION  (RGB-based, no IR sensor)
# ──────────────────────────────────────────────

class LivenessDetector:
    """
    Detects that a live face — not a photo or screen replay — is present.
    Uses eye-aspect-ratio (EAR) blink detection across a frame sequence.
    """

    EAR_BLINK_THRESHOLD = 0.25   # below this = eye closed
    MIN_BLINK_FRAMES    = 2      # consecutive closed frames = confirmed blink

    def __init__(self):
        self._closed_count = 0
        self.blink_detected = False

    def _eye_aspect_ratio(self, eye_landmarks: list) -> float:
        # Vertical distances
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        # Horizontal distance
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def update(self, frame_rgb: np.ndarray) -> bool:
        """Feed one frame; returns True once liveness is confirmed."""
        if self.blink_detected:
            return True

        landmarks_list = face_recognition.face_landmarks(frame_rgb)
        if not landmarks_list:
            return False

        lm = landmarks_list[0]
        left_ear  = self._eye_aspect_ratio(lm.get("left_eye",  []))
        right_ear = self._eye_aspect_ratio(lm.get("right_eye", []))
        ear       = (left_ear + right_ear) / 2.0

        if ear < self.EAR_BLINK_THRESHOLD:
            self._closed_count += 1
        else:
            if self._closed_count >= self.MIN_BLINK_FRAMES:
                self.blink_detected = True
            self._closed_count = 0

        return self.blink_detected

    def reset(self):
        self._closed_count  = 0
        self.blink_detected = False


# ──────────────────────────────────────────────
# FACE PIPELINE  (detection → alignment → embedding)
# ──────────────────────────────────────────────

class FacePipeline:
    """
    Detects a face, extracts a 128-dim embedding via dlib (face_recognition).
    Embeddings are L2-normalised so comparison == cosine similarity.
    """

    def extract_embedding(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(frame_rgb, model="hog")
        if not locations:
            return None
        encodings = face_recognition.face_encodings(frame_rgb, locations)
        if not encodings:
            return None
        vec = np.array(encodings[0])
        return vec / np.linalg.norm(vec)   # L2 normalise

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))         # both already unit vectors


# ──────────────────────────────────────────────
# ENROLLMENT
# ──────────────────────────────────────────────

def enroll(camera_index: int = 0) -> None:
    """Capture ENROLLMENT_FRAMES face embeddings and persist to Keychain."""
    print("\n[ENROLL] Look at the camera. Slightly turn your head left and right.")
    store    = SecureStore()
    pipeline = FacePipeline()
    cap      = cv2.VideoCapture(camera_index)

    embeddings: list[np.ndarray] = []
    attempt = 0

    while len(embeddings) < ENROLLMENT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        attempt += 1
        embedding = pipeline.extract_embedding(frame)

        cv2.putText(frame, f"Captured: {len(embeddings)}/{ENROLLMENT_FRAMES}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.imshow("Enrollment — press Q to cancel", frame)

        if embedding is not None:
            embeddings.append(embedding)
            print(f"  Frame {len(embeddings)}/{ENROLLMENT_FRAMES} captured.")
            time.sleep(0.3)   # slight pause so frames aren't identical

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[ENROLL] Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    store.save(embeddings)
    print(f"[ENROLL] Done. {len(embeddings)} embeddings saved to Keychain.")


# ──────────────────────────────────────────────
# AUTHENTICATION
# ──────────────────────────────────────────────

class FaceAuthenticator:

    def __init__(self, camera_index: int = 0):
        self.store     = SecureStore()
        self.pipeline  = FacePipeline()
        self.camera    = camera_index
        self._failures = 0
        self._locked_until: float = 0.0

    def _is_locked_out(self) -> bool:
        if self._failures >= MAX_AUTH_ATTEMPTS:
            remaining = self._locked_until - time.time()
            if remaining > 0:
                print(f"[AUTH] Locked out. Try again in {remaining:.0f}s.")
                return True
            self._failures = 0   # lockout expired
        return False

    def authenticate(self) -> bool:
        if self._is_locked_out():
            return False

        stored = self.store.load()
        if stored is None:
            print("[AUTH] No face enrolled. Run enroll() first.")
            return False

        print("[AUTH] Look at the camera…")
        cap      = cv2.VideoCapture(self.camera)
        liveness = LivenessDetector()
        start    = time.time()
        timeout  = 15   # seconds

        matched = False

        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Liveness check
            if LIVENESS_BLINK_REQUIRED:
                liveness.update(frame_rgb)

            embedding = self.pipeline.extract_embedding(frame)

            status_text = "Blink to confirm liveness…" if (
                LIVENESS_BLINK_REQUIRED and not liveness.blink_detected
            ) else "Scanning…"

            cv2.putText(frame, status_text,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("Face ID — press Q to cancel", frame)

            if embedding is not None:
                similarities = [self.pipeline.cosine_similarity(embedding, s) for s in stored]
                best = max(similarities)

                if best >= SIMILARITY_THRESHOLD:
                    liveness_ok = (not LIVENESS_BLINK_REQUIRED) or liveness.blink_detected
                    if liveness_ok:
                        matched = True
                        break
                    else:
                        cv2.putText(frame, "Liveness required — blink please",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if matched:
            self._failures = 0
            print("[AUTH] Authenticated.")
            return True
        else:
            self._failures += 1
            remaining = MAX_AUTH_ATTEMPTS - self._failures
            if remaining <= 0:
                self._locked_until = time.time() + LOCKOUT_SECONDS
                print(f"[AUTH] Too many failures. Locked for {LOCKOUT_SECONDS}s.")
            else:
                print(f"[AUTH] Not recognised. {remaining} attempt(s) left.")
            return False


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "auth"

    if cmd == "enroll":
        enroll()
    elif cmd == "auth":
        auth = FaceAuthenticator()
        result = auth.authenticate()
        sys.exit(0 if result else 1)
    elif cmd == "reset":
        SecureStore().clear()
        print("[RESET] All face data deleted from Keychain.")
    else:
        print("Usage: python face_id_recognition.py [enroll | auth | reset]")
        sys.exit(1)
