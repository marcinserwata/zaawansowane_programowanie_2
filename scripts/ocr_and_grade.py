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

def detect_plate_angle(crop):
    """
    Detects the rotation angle of the license plate using edge detection and Hough lines.
    Returns the angle in degrees needed to make the plate horizontal.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 70, apertureSize=3)  # TUNE: Canny thresholds (50, 150) - lower values detect more edges
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)  # TUNE: threshold=50 (line detection sensitivity), minLineLength=30 (minimum line length), maxLineGap=10 (max gap in line)
    
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize angle to [-45, 45] range
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)
    
    if not angles:
        return 0
    
    # Return median angle for robustness
    return np.median(angles)

def rotate_image(image, angle):
    """
    Rotates the image by the given angle around its center.
    """
    if abs(angle) < 0.5:  # TUNE: Skip rotation threshold (0.5 degrees) - increase to ignore smaller rotations
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def correct_perspective(crop):
    """
    Corrects perspective distortion by finding the license plate contour
    and applying perspective transformation to make it rectangular.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # TUNE: Canny thresholds (50, 150) - adjust for better edge detection
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return crop
    
    # Find the largest contour (assumed to be the plate)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # TUNE: epsilon factor (0.02) - lower values = more precise approximation
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we found a quadrilateral, apply perspective correction
    if len(approx) == 4:
        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # Sort points to get consistent ordering
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left point has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Top-right has smallest diff, bottom-left has largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        # Calculate dimensions for the corrected rectangle
        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(rect[3] - rect[0])
        height_right = np.linalg.norm(rect[2] - rect[1])
        height = int(max(height_left, height_right))
        
        # Define destination points for rectangle
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(rect, dst)
        corrected = cv2.warpPerspective(crop, matrix, (width, height))
        
        return corrected
    
    return crop

def process_detected_plate(crop, file_name):
    if crop.size == 0:
        return ""    
    
    # Step 1: Correct perspective distortion
    crop = correct_perspective(crop)    
    
    # Step 2: Detect and correct plate angle
    angle = detect_plate_angle(crop)
    if abs(angle) > 0.5:
        crop = rotate_image(crop, angle)        
    
    # Step 3: Scale image for better OCR
    h, w = crop.shape[:2]
    scale_factor = 3  # TUNE: Scale factor (3) - higher values = larger text for OCR
    crop = cv2.resize(crop, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)    
    
    # Step 4: Convert to grayscale and blur
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)  # TUNE: kernel size (3) - larger values = more blur, must be odd
    cv2.imwrite(str(CROPS_DIR / f"{Path(file_name).stem}_final.jpg"), blurred)
    
    # Apply OCR to the processed crop
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(blurred, config=config)
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
