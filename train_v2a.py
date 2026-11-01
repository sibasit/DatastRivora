
from pathlib import Path
from ultralytics import YOLO
import torch

# ==========================================
# 1. KONFIGURASI PROJECT
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_YAML = (
    ROOT
    / "dataset"
    / "rivora.v1-rivora_detect.yolov8"
    / "data.yaml"
)

OUTPUT_DIR = ROOT / "runs" / "detect"

# ==========================================
# 2. VALIDASI AWAL
# ==========================================

def check_environment():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml tidak ditemukan: {DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA tidak tersedia. Periksa instalasi PyTorch."
        )

    print("====================================")
    print("       RIVORA V2-A TRAINING")
    print("====================================")

    print("Dataset:", DATA_YAML)
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)

    print("Image Size: 800")
    print("Batch Size: 2")
    print("Epoch: 100")

    print("====================================")


# ==========================================
# 3. TRAINING V2-A
# ==========================================

def main():

    check_environment()

    # Gunakan pretrained YOLOv8n yang sama
    # dengan eksperimen baseline V1.
    model = YOLO("yolov8n.pt")

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Perubahan eksperimen V2-A
        imgsz=800,

        # Parameter training
        epochs=100,
        batch=2,
        patience=20,

        # Hardware
        device=0,
        workers=0,
        amp=True,
        cache=False,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Output
        project=str(OUTPUT_DIR),
        name="rivora_v2a_img800",
        exist_ok=False,
        save=True,
        plots=True,
        verbose=True,

        # Tidak mengubah augmentasi dari default
        # Ultralytics yang digunakan pada V1.
    )

    print("\n====================================")
    print("       TRAINING V2-A SELESAI")
    print("====================================")

    print("Periksa folder:")
    print(
        OUTPUT_DIR / "rivora_v2a_img800"
    )

    print("\nModel terbaik:")
    print(
        OUTPUT_DIR
        / "rivora_v2a_img800"
        / "weights"
        / "best.pt"
    )

    print("\nModel terakhir:")
    print(
        OUTPUT_DIR
        / "rivora_v2a_img800"
        / "weights"
        / "last.pt"
    )


if __name__ == "__main__":
    main()
