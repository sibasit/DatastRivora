
from ultralytics import YOLO
from pathlib import Path
import torch

# ==========================================
# RIVORA - YOLOv8 OBJECT DETECTION
# Version: V1 Baseline
# ==========================================

def main():

    print("=" * 50)
    print("       RIVORA OBJECT DETECTION")
    print("       YOLOv8 TRAINING")
    print("=" * 50)

    # 1. Lokasi project
    ROOT = Path(__file__).resolve().parent

    DATA_YAML = (
        ROOT
        / "dataset"
        / "rivora.v1-rivora_detect.yolov8"
        / "data.yaml"
    )

    # 2. Periksa dataset
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_YAML}"
        )

    print("\n[1] Dataset ditemukan!")

    # 3. Periksa GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU CUDA tidak tersedia. "
            "Periksa instalasi PyTorch."
        )

    print("[2] GPU:", torch.cuda.get_device_name(0))

    # 4. Load pretrained YOLOv8 Nano
    print("\n[3] Loading YOLOv8 Nano...")

    model = YOLO("yolov8n.pt")

    # 5. Mulai training
    print("\n[4] Memulai training RIVORA V1...")

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Model training
        epochs=100,
        imgsz=640,
        batch=2,

        # GPU
        device=0,
        workers=0,

        # Optimasi training
        patience=20,
        amp=True,
        cache=False,

        # Reproducibility
        seed=42,

        # Output
        project=str(ROOT / "runs" / "detect"),
        name="rivora_v1_yolov8n",
        plots=True,
        save=True,

    )

    print("\n" + "=" * 50)
    print("       TRAINING RIVORA SELESAI!")
    print("=" * 50)

    print(
        "Best Model:",
        ROOT / "runs" / "detect" /
        "rivora_v1_yolov8n" / "weights" / "best.pt"
    )


if __name__ == "__main__":
    main()
