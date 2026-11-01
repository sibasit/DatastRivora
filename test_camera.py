
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "rivora_v2a_img800"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "Sampah Anorganik",
    1: "Sampah Organik"
}

# Confidence awal dari F1 curve V2-A
CONFIDENCE = 0.30

IMAGE_SIZE = 800

# ==========================================
# 2. LOAD MODEL
# ==========================================

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    print("====================================")
    print("       RIVORA REAL-TIME CAMERA")
    print("====================================")

    device = 0 if torch.cuda.is_available() else "cpu"

    if device == 0:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU tidak tersedia, menggunakan CPU.")

    model = YOLO(str(MODEL_PATH))

    print("Model berhasil dimuat!")
    print("Model:", MODEL_PATH)

    # ======================================
    # 3. BUKA WEBCAM
    # ======================================

    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            "Kamera tidak dapat dibuka. "
            "Coba ganti index kamera menjadi 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nKamera berhasil dibuka.")
    print("Tekan Q untuk keluar.")

    # ======================================
    # 4. REAL-TIME DETECTION
    # ======================================

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # Prediksi objek
        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=device,
            verbose=False
        )

        result = results[0]

        # Salin frame asli untuk menggambar hasil
        output = frame.copy()

        counts = {
            0: 0,
            1: 0
        }

        if result.boxes is not None:

            for box in result.boxes:

                # Koordinat bounding box
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                # ID kelas dan confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}"
                )

                counts[class_id] = (
                    counts.get(class_id, 0) + 1
                )

                # Warna bounding box (BGR)
                if class_id == 0:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)

                # Gambar bounding box
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Label
                label = (
                    f"{class_name}: {confidence:.2f}"
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ==================================
        # 5. TAMPILKAN INFORMASI DETEKSI
        # ==================================

        cv2.rectangle(
            output,
            (10, 10),
            (350, 115),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            output,
            "RIVORA - RIVER WASTE DETECTION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Anorganik: {counts[0]}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 100),
            2
        )

        cv2.putText(
            output,
            f"Organik: {counts[1]}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "RIVORA - Real Time Detection",
            output
        )

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ======================================
    # 6. TUTUP KAMERA
    # ======================================

    camera.release()
    cv2.destroyAllWindows()

    print("\nRIVORA Camera Testing selesai.")


if __name__ == "__main__":
    main()
