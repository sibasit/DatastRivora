
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "rivora.v1-rivora_detect.yolov8"

splits = ["train", "valid", "test"]
class_names = {
    0: "Sampah Anorganik",
    1: "Sampah Organik"
}

for split in splits:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    image_count = len(list(image_dir.glob("*")))
    label_files = list(label_dir.glob("*.txt"))

    counts = Counter()
    invalid = []

    for file in label_files:
        for line_no, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                invalid.append((file.name, line_no, "Format salah"))
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid.append((file.name, line_no, "Nilai bukan angka"))
                continue

            if cls not in class_names:
                invalid.append((file.name, line_no, "ID kelas tidak dikenal"))
                continue

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                invalid.append((file.name, line_no, "Koordinat di luar rentang"))

            if w <= 0 or h <= 0:
                invalid.append((file.name, line_no, "Ukuran bbox tidak valid"))

            counts[cls] += 1

    print(f"\n===== {split.upper()} =====")
    print("Jumlah gambar:", image_count)
    print("Jumlah file label:", len(label_files))

    for cls, name in class_names.items():
        print(f"{name}: {counts[cls]} anotasi")

    print("Jumlah anotasi:", sum(counts.values()))
    print("Label bermasalah:", len(invalid))

    for item in invalid[:5]:
        print("  ", item)
