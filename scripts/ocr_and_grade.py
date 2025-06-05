import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
Skrypt wykonuje inferencję wytrenowanym modelem YOLO11, wycina detekcje tablic rejestracyjnych,
przeprowadza OCR za pomocą Tesseract, liczy accuracy, czas przetwarzania oraz wylicza ocenę końcową.
"""

SCRIPT_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT    = SCRIPT_DIR.parent
WEIGHTS_PATH    = PROJECT_ROOT / "runs" / "plate_detection_yolov11" / "weights" / "best.pt"
VAL_IMAGES_DIR  = PROJECT_ROOT / "yolo_data" / "images" / "val"
CROPS_DIR       = PROJECT_ROOT / "results" / "crops"
OCR_RESULTS     = PROJECT_ROOT / "results" / "ocr_results.txt"
# Plik, w którym zapiszemy tylko błędne dopasowania OCR:
BAD_MATCHES_FILE = PROJECT_ROOT / "results" / "ocr_bad_matches.txt"
ANNOTATIONS_XML = PROJECT_ROOT / "data" / "annotations.xml"

def calculate_final_grade(accuracy_percent: float, time_for_100: float) -> float:
    """
    Wylicza ocenę końcową według wzoru:
      - jeśli accuracy < 60% lub time_for_100 > 60s => 2.0
      - w przeciwnym razie:
          accuracy_norm = (accuracy - 60) / 40
          time_norm     = (60 - time_for_100) / 50
          score         = 0.7 * accuracy_norm + 0.3 * time_norm
          grade         = 2.0 + 3.0 * score
          zwraca grade zaokrąglone do najbliższej 0.5
    """
    if accuracy_percent < 60 or time_for_100 > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm     = (60 - time_for_100) / 50
    score         = 0.7 * accuracy_norm + 0.3 * time_norm
    grade         = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def load_annotations(xml_path: Path) -> dict:
    """
    Parsuje plik annotations.xml i zwraca słownik: {nazwa_pliku.jpg: {'plate_number': str, 'bbox': (xmin, ymin, xmax, ymax)}}.    
    """
    if not xml_path.is_file():
        raise FileNotFoundError(f"[ERROR] Nie znaleziono pliku adnotacji: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    gt = {}
    for image in root.findall("image"):
        file_name = image.attrib["name"]
        for box in image.findall("box"):
            plate_attr = box.find("attribute[@name='plate number']")
            if plate_attr is not None:
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])
                gt[file_name] = {
                    'plate_number': plate_attr.text.strip(),
                    'bbox': (xtl, ytl, xbr, ybr)
                }
                break
    return gt

def calculate_iou(box1, box2):
    """
    Oblicza IoU (Intersection over Union) między dwoma bounding boxami.
    Każdy box to krotka (xmin, ymin, xmax, ymax).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2    
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)    
    
    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        intersection = 0.0
    else:
        intersection = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)    
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)    
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def process_detected_plate(crop, file_name):
    if crop.size == 0:
        return ""
    
    # Zapisz wycinek oryginalny (do weryfikacji)
    cv2.imwrite(str(CROPS_DIR / f"{Path(file_name).stem}_cropped.jpg"), crop)
    
    # Skalowanie i przetwarzanie w jednym etapie
    h, w = crop.shape[:2]
    crop_scaled = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_scaled, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.medianBlur(denoised, 5)
    
    # Progowanie adaptacyjne z morfologią
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(str(CROPS_DIR / f"processed_{Path(file_name).stem}.jpg"), thresh)
    
    # OCR
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.replace(" ", "").replace("\n", "").upper()

def detection_and_ocr():    
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"[ERROR] Nie znaleziono wag modelu: {WEIGHTS_PATH}\n"            
        )
    
    if not VAL_IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"[ERROR] Nie znaleziono folderu z obrazami walidacyjnymi: {VAL_IMAGES_DIR}"
        )
    
    annotations = load_annotations(ANNOTATIONS_XML)
    
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Ładowanie modelu z wagami: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))
    
    val_files = sorted([f.name for f in VAL_IMAGES_DIR.iterdir() if f.suffix.lower() == ".jpg"])
    if len(val_files) == 0:
        raise RuntimeError(f"[ERROR] Brak plików .jpg w katalogu walidacyjnym: {VAL_IMAGES_DIR}")
    
    results = []
    processing_times = []
    ious = []

    for file_name in tqdm(val_files[:100], desc="Detekcja tablic i OCR"):
        img_path = VAL_IMAGES_DIR / file_name
        img_bgr = cv2.imread(str(img_path))
        
        start = time.time()
                
        detections = model(str(img_path))[0].boxes.data
        detected_bbox = None
        if detections is None or len(detections) == 0:
            ocr_plate_num = ""
        else:            
            best_det = detections[np.argmax(detections[:, 4].cpu().numpy())]
            xmin, ymin, xmax, ymax = map(int, best_det[:4].cpu().numpy())
            detected_bbox = (map(int, best_det[:4].cpu().numpy()))
            
            plate_w = xmax - xmin
            new_xmin = min(max(xmin + int(0.09 * plate_w), xmin), xmax)
            crop = img_bgr[ymin:ymax, new_xmin:xmax]            
            
            ocr_plate_num = process_detected_plate(crop, file_name)
        
        processing_times.append(time.time() - start)        
        
        iou = 0.0
        annotation = annotations.get(file_name, {})
        if annotation and detected_bbox is not None:
            annotation_bbox = annotation['bbox']
            iou = calculate_iou(detected_bbox, annotation_bbox)
        ious.append(iou)
        
        expected_plate_num = annotation.get('plate_number', '').upper() if annotation else ''

        results.append({
            "file_name": file_name,
            "ocr_plate_num": ocr_plate_num,
            "expected_plate_num": expected_plate_num,
            "iou": iou
        })
    
    valid_results = [r for r in results if r["expected_plate_num"]]
    correct = sum(1 for r in valid_results if r["ocr_plate_num"] == r["expected_plate_num"])
    accuracy_percent = 100.0 * correct / len(valid_results) if valid_results else 0.0    
    
    valid_ious = [iou for iou in ious if iou > 0]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    num_processed = len(results)
    total_time = sum(processing_times)
    avg_time = total_time / num_processed if num_processed > 0 else 0.0    
    time_for_100 = total_time * (100 / num_processed) if num_processed > 0 else float('inf')
    
    final_grade = calculate_final_grade(accuracy_percent, time_for_100)
    
    results.sort(key=lambda x: x["file_name"])
    
    with open(str(OCR_RESULTS), "w", encoding="utf-8") as f:
        f.write("file_name,expected_plate_num,ocr_plate_num,iou\n")
        for r in results:
            f.write(f"{r['file_name']},{r['expected_plate_num']},{r['ocr_plate_num']},{r['iou']:.4f}\n")

    # 12.5 Zapis błędnych dopasowań do osobnego pliku
    with open(str(BAD_MATCHES_FILE), "w", encoding="utf-8") as f:
        f.write("file_name,expected_plate_num,ocr_plate_num,iou\n")
        for r in results:
            if r["expected_plate_num"] and r["ocr_plate_num"] != r["expected_plate_num"]:
                f.write(f"{r['file_name']},{r['expected_plate_num']},{r['ocr_plate_num']},{r['iou']:.4f}\n")
    
    print("\n ------------[WYNIK]------------")
    print(f"\nPrzetworzono {num_processed} obrazów.")
    print(f"Liczba obrazów z GT: {len(valid_results)}")
    print(f"(poprawnie odczytanych / z GT): {correct} / {len(valid_results)}")
    print(f"Accuracy OCR: {accuracy_percent:.2f}%")
    print(f"Średni IoU: {mean_iou:.4f}")
    print(f"Liczba detekcji z IoU > 0.7: {sum(1 for iou in ious if iou > 0.7)}")
    print(f"Łączny czas przetwarzania {num_processed} obrazów: {total_time:.2f}s")
    print(f"Śr. czas na obraz: {avg_time:.3f}s")
    print(f"Ekstrapolowany czas na 100 obrazów: {time_for_100:.2f}s")
    print(f"Ostateczna ocena końcowa: {final_grade:.1f}")
    print("\n ------------[WYNIK]------------")

if __name__ == "__main__":
    detection_and_ocr()
