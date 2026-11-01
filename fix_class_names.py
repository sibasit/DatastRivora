
from ultralytics import YOLO
from pathlib import Path

# Lokasi project
ROOT = Path(__file__).resolve().parent

# Lokasi model hasil training
MODEL_PATH = ROOT / "runs" / "detect" / "rivora_v1_yolov8n" / "weights" / "best.pt"

# Lokasi output model baru
OUTPUT_PATH = ROOT / "runs" / "detect" / "rivora_v1_yolov8n" / "weights" / "best_named.pt"

def main():
    model = YOLO(str(MODEL_PATH))

    # Ubah nama kelas pada model internal
    model.model.names = {
        0: "Sampah Anorganik",
        1: "Sampah Organik"
    }

    # Simpan sebagai model baru
    model.save(str(OUTPUT_PATH))

    # Periksa hasilnya
    check = YOLO(str(OUTPUT_PATH))

    print("\nNama kelas berhasil diperbarui!")
    print("Nama kelas:", check.names)
    print("Model tersimpan di:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
