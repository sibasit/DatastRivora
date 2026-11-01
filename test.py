
from ultralytics import YOLO
from pathlib import Path


def main():

    print("=" * 50)
    print("       RIVORA - MODEL EVALUATION")
    print("=" * 50)

    # Lokasi project
    ROOT = Path(__file__).resolve().parent

    # Lokasi dataset
    DATA_YAML = (
        ROOT
        / "dataset"
        / "rivora.v1-rivora_detect.yolov8"
        / "data.yaml"
    )

    # Lokasi model terbaik
    MODEL_PATH = (
        ROOT
        / "runs"
        / "detect"
        / "rivora_v1_yolov8n"
        / "weights"
        / "best.pt"
    )

    # Periksa file
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_YAML}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    print("\n[1] Dataset ditemukan")
    print("[2] Best model ditemukan")

    # Load model
    model = YOLO(str(MODEL_PATH))

    print("\n[3] Memulai evaluasi TEST DATASET...")

    # Evaluasi menggunakan test set
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / "runs" / "detect"),
        name="rivora_v1_test",
        exist_ok=True
    )

    print("\n" + "=" * 50)
    print("       RIVORA TEST RESULTS")
    print("=" * 50)

    print(f"Precision : {results.box.mp:.4f}")
    print(f"Recall    : {results.box.mr:.4f}")
    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")

    print("\nEvaluasi berhasil diselesaikan!")


if __name__ == "__main__":
    main()
