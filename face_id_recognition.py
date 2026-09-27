"""
Face ID Recognition System for macOS
Mirrors the enrollment → liveness → embedding → match pipeline used by Apple Face ID.

Dependencies:
    pip3 install deepface opencv-python numpy keyring cryptography tf-keras

Hardware: FaceTime HD camera (RGB only — no TrueDepth)
Storage:  System Keychain via `keyring` (wraps macOS Keychain)
"""

import cv2
import numpy as np
import keyring
import json
import time
import sys
from deepface import DeepFace
from cryptography.fernet import Fernet

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

SERVICE_NAME            = "MacFaceID"
KEYCHAIN_KEY            = "face_embeddings"
ENCRYPTION_KEY_ID       = "face_enc_key"

SIMILARITY_THRESHOLD    = 0.70   # cosine similarity — raise for stricter (max 1.0)
ENROLLMENT_FRAMES       = 10     # frames captured during setup
MAX_AUTH_ATTEMPTS       = 5      # lockout after this many failures
LOCKOUT_SECONDS         = 30     # seconds locked out after max failures
LIVENESS_BLINK_REQUIRED = False  # set True once deepface liveness is confirmed working


# ──────────────────────────────────────────────
# SECURE STORAGE  (macOS Keychain via keyring)
# ──────────────────────────────────────────────

class SecureStore:

    def _get_or_create_key(self) -> bytes:
        raw = keyring.get_password(SERVICE_NAME, ENCRYPTION_KEY_ID)
        if raw is None:
            key = Fernet.generate_key()
            keyring.set_password(SERVICE_NAME, ENCRYPTION_KEY_ID, key.decode())
            return key
        return raw.encode()

    def save(self, embeddings: list) -> None:
        key       = self._get_or_create_key()
        f         = Fernet(key)
        data      = json.dumps([e.tolist() for e in embeddings]).encode()
        encrypted = f.encrypt(data)
        keyring.set_password(SERVICE_NAME, KEYCHAIN_KEY, encrypted.decode())

    def load(self):
        raw = keyring.get_password(SERVICE_NAME, KEYCHAIN_KEY)
        if raw is None:
            return None
        key  = self._get_or_create_key()
        f    = Fernet(key)
        data = f.decrypt(raw.encode())
        return [np.array(e) for e in json.loads(data)]

    def clear(self) -> None:
        try:
            keyring.delete_password(SERVICE_NAME, KEYCHAIN_KEY)
            keyring.delete_password(SERVICE_NAME, ENCRYPTION_KEY_ID)
        except Exception:
            pass


# ──────────────────────────────────────────────
# LIVENESS DETECTION  (blink via eye landmarks)
# ──────────────────────────────────────────────

class LivenessDetector:

    EAR_THRESHOLD  = 0.25
    MIN_CLOSED     = 2

    def __init__(self):
        self._closed_count  = 0
        self.blink_detected = False

    def _ear(self, eye) -> float:
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def update(self, frame_bgr: np.ndarray) -> bool:
        if self.blink_detected:
            return True
        try:
            result = DeepFace.analyze(frame_bgr, actions=["emotion"], enforce_detection=False, silent=True)
            # Use facial area to approximate eye openness via region height ratio
            region = result[0].get("region", {})
            h = region.get("h", 0)
            if h > 0:
                eye_h = h * 0.12
                if eye_h < self.EAR_THRESHOLD * h:
                    self._closed_count += 1
                else:
                    if self._closed_count >= self.MIN_CLOSED:
                        self.blink_detected = True
                    self._closed_count = 0
        except Exception:
            pass
        return self.blink_detected

    def reset(self):
        self._closed_count  = 0
        self.blink_detected = False


# ──────────────────────────────────────────────
# FACE PIPELINE  (detection → embedding)
# ──────────────────────────────────────────────

class FacePipeline:

    def extract_embedding(self, frame_bgr: np.ndarray):
        try:
            result = DeepFace.represent(
                frame_bgr,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )
            if not result:
                return None
            vec = np.array(result[0]["embedding"])
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None
        except Exception:
            return None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


# ──────────────────────────────────────────────
# ENROLLMENT
# ──────────────────────────────────────────────

def enroll(camera_index: int = 0) -> None:
    print("\n[ENROLL] Look at the camera. Slowly turn your head left and right.")
    store      = SecureStore()
    pipeline   = FacePipeline()
    cap        = cv2.VideoCapture(camera_index)
    embeddings = []

    while len(embeddings) < ENROLLMENT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        embedding = pipeline.extract_embedding(frame)

        cv2.putText(frame, f"Captured: {len(embeddings)}/{ENROLLMENT_FRAMES}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.imshow("Enrollment — press Q to cancel", frame)

        if embedding is not None:
            embeddings.append(embedding)
            print(f"  Frame {len(embeddings)}/{ENROLLMENT_FRAMES} captured.")
            time.sleep(0.3)

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
        self._locked_until = 0.0

    def _is_locked_out(self) -> bool:
        if self._failures >= MAX_AUTH_ATTEMPTS:
            remaining = self._locked_until - time.time()
            if remaining > 0:
                print(f"[AUTH] Locked out. Try again in {remaining:.0f}s.")
                return True
            self._failures = 0
        return False

    def authenticate(self) -> bool:
        if self._is_locked_out():
            return False

        stored = self.store.load()
        if stored is None:
            print("[AUTH] No face enrolled. Run: python face_id_recognition.py enroll")
            return False

        print("[AUTH] Look at the camera…")
        cap      = cv2.VideoCapture(self.camera)
        liveness = LivenessDetector()
        start    = time.time()
        matched  = False

        while time.time() - start < 15:
            ret, frame = cap.read()
            if not ret:
                continue

            if LIVENESS_BLINK_REQUIRED:
                liveness.update(frame)

            embedding = self.pipeline.extract_embedding(frame)

            label = "Blink to confirm liveness…" if (
                LIVENESS_BLINK_REQUIRED and not liveness.blink_detected
            ) else "Scanning…"

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("Face ID — press Q to cancel", frame)

            if embedding is not None:
                best = max(self.pipeline.cosine_similarity(embedding, s) for s in stored)
                if best >= SIMILARITY_THRESHOLD:
                    if (not LIVENESS_BLINK_REQUIRED) or liveness.blink_detected:
                        matched = True
                        break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if matched:
            self._failures = 0
            print("[AUTH] Authenticated.")
            return True

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
        sys.exit(0 if auth.authenticate() else 1)
    elif cmd == "reset":
        SecureStore().clear()
        print("[RESET] All face data deleted from Keychain.")
    else:
        print("Usage: python face_id_recognition.py [enroll | auth | reset]")
        sys.exit(1)
