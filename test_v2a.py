
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_YAML = (
    ROOT / "dataset"
    / "rivora.v1-rivora_detect.yolov8"
    / "data.yaml"
)

MODEL_PATH = (
    ROOT / "runs" / "detect"
    / "rivora_v2a_img800"
    / "weights" / "best.pt"
)

OUTPUT_DIR = ROOT / "runs" / "detect"


# ==========================================
# 2. TESTING
# ==========================================

def main():

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("====================================")
    print("       RIVORA V2-A TESTING")
    print("====================================")

    print("Model:", MODEL_PATH)
    print("Dataset:", DATA_YAML)

    model = YOLO(str(MODEL_PATH))

    # Evaluasi menggunakan test split
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=800,
        batch=2,
        device=0,
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="rivora_v2a_test",
        exist_ok=False,
        verbose=True
    )

    # ======================================
    # 3. TAMPILKAN HASIL EVALUASI
    # ======================================

    print("\n====================================")
    print("       HASIL TEST RIVORA V2-A")
    print("====================================")

    metrics = results.box

    print(f"Precision : {metrics.mp:.4f}")
    print(f"Recall    : {metrics.mr:.4f}")
    print(f"mAP50     : {metrics.map50:.4f}")
    print(f"mAP50-95  : {metrics.map:.4f}")

    print("\nAP50 per kelas:")

    class_names = {
        0: "Sampah Anorganik",
        1: "Sampah Organik"
    }

    for idx, ap in enumerate(metrics.ap50):
        print(f"{class_names[idx]}: {ap:.4f}")

    print("\nHasil disimpan di:")
    print(OUTPUT_DIR / "rivora_v2a_test")


if __name__ == "__main__":
    main()
