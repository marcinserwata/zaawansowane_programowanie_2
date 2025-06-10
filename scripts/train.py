import os
from ultralytics import YOLO

SCRIPT_DIR    = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_YAML    = os.path.join(PROJECT_ROOT, "yolo_data", "plate_dataset.yaml")
PRETRAINED   = os.path.join(PROJECT_ROOT, "yolo11n.pt")
IMGSZ        = 640
BATCH_SIZE   = 16
EPOCHS       = 20
PROJECT_DIR  = os.path.join(PROJECT_ROOT, "runs")
RUN_NAME     = "plate_detection_yolov11"
DEVICE       = "0"

def train():    
    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError(f"[ERROR] Nie znaleziono pliku z opisem datasetu: {DATA_YAML}")
    
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    model = YOLO(PRETRAINED)

    print("Rozpoczynam trenowanie...")
    model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=DEVICE
    )

    print("Trenowanie zakończone.")
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    last_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")
    print(f"Najlepszy model → {best_weights}")
    print(f"Ostateczny model → {last_weights}")

if __name__ == "__main__":
    train()
