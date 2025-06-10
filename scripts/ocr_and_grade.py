import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SCRIPT_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT    = SCRIPT_DIR.parent
WEIGHTS_PATH    = PROJECT_ROOT / "runs" / "plate_detection_yolov11" / "weights" / "best.pt"
VAL_IMAGES_DIR  = PROJECT_ROOT / "yolo_data" / "images" / "val"
CROPS_DIR       = PROJECT_ROOT / "results" / "crops"
OCR_RESULTS     = PROJECT_ROOT / "results" / "ocr_results.txt"
ANNOTATIONS_XML = PROJECT_ROOT / "data" / "annotations.xml"

def calculate_final_grade(accuracy_percent: float, time_for_100: float) -> float:
    """Oblicza końcową ocenę na podstawie dokładności i czasu przetwarzania"""
    if accuracy_percent < 60 or time_for_100 > 60:
        return 2.0    
    
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm     = (60 - time_for_100) / 50    
    
    score         = 0.7 * accuracy_norm + 0.3 * time_norm
    grade         = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def load_annotations(xml_path: Path) -> dict:
    """Wczytuje adnotacje z pliku XML zawierającego prawdziwe numery tablic"""
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
                # Współrzędne bounding box
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
    """Oblicza IoU (Intersection over Union) między dwoma bounding boxami"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2    
    
    # Współrzędne przecięcia
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)    
    
    # Obliczenie powierzchni przecięcia
    if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
        intersection = 0.0
    else:
        intersection = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)    
    
    # Obliczenie powierzchni obu boxów i ich sumy
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)    
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def detect_plate_angle(crop):
    """Wykrywa kąt nachylenia tablicy rejestracyjnej"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 70, apertureSize=3)
    
    # Detekcja linii za pomocą transformaty Hougha
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return 0
    
    # Obliczenie kątów wszystkich linii
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalizacja kątów do zakresu -45° do 45°
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)
    
    if not angles:
        return 0
    
    return np.median(angles)

def rotate_image(image, angle):
    """Obraca obraz o zadany kąt"""
    if abs(angle) < 0.5:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Macierz rotacji i zastosowanie transformacji
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def correct_perspective(crop):
    """Koryguje perspektywę tablicy rejestracyjnej"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Znajdowanie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return crop
    
    # Wybór największego konturu
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Aproksymacja konturu do czworokąta
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # Sortowanie punktów: lewy górny, prawy górny, prawy dolny, lewy dolny
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # lewy górny
        rect[2] = pts[np.argmax(s)]      # prawy dolny
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # prawy górny
        rect[3] = pts[np.argmax(diff)]   # lewy dolny
        
        # Obliczenie wymiarów docelowego prostokąta
        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(rect[3] - rect[0])
        height_right = np.linalg.norm(rect[2] - rect[1])
        height = int(max(height_left, height_right))
        
        # Punkty docelowe dla korekcji perspektywy
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Zastosowanie korekcji perspektywy
        matrix = cv2.getPerspectiveTransform(rect, dst)
        corrected = cv2.warpPerspective(crop, matrix, (width, height))
        
        return corrected
    
    return crop

def correct_plate_pattern(text):
    """Koryguje wzorzec tablicy - pierwsze 2 znaki to litery, reszta to cyfry"""
    if len(text) < 2:
        return text
    
    corrected = ""
    for i, char in enumerate(text):
        if i < 2:  # Pierwsze dwa znaki powinny być literami
            if char.isdigit():
                # Mapowanie cyfr podobnych do liter
                digit_to_letter = {
                    '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', 
                    '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
                }
                corrected += digit_to_letter.get(char, 'O')
            else:
                corrected += char
        else:
            corrected += char
    
    return corrected

def process_detected_plate(crop, file_name):
    """Przetwarza wykrytą tablicę rejestracyjną i wykonuje OCR"""
    if crop.size == 0:
        return ""    
    
    # Korekcja perspektywy
    crop = correct_perspective(crop)    
    
    # Korekcja rotacji
    angle = detect_plate_angle(crop)
    if abs(angle) > 0.5:
        crop = rotate_image(crop, angle)        
    
    # Skalowanie obrazu dla lepszej jakości OCR
    h, w = crop.shape[:2]
    scale_factor = 3
    crop = cv2.resize(crop, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)    
    
    # Przetwarzanie obrazu przed OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    
    # Konfiguracja Tesseract dla tablic rejestracyjnych
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(blurred, config=config)
    cleaned_text = text.replace(" ", "").replace("\n", "").upper()
    
    # Korekcja wzorca tablicy
    corrected_text = correct_plate_pattern(cleaned_text)
    
    return corrected_text

def detection_and_ocr():
    """Główna funkcja wykonująca detekcję tablic i OCR"""
    # Sprawdzenie istnienia wymaganych plików i katalogów
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
    
    model = YOLO(str(WEIGHTS_PATH))    
    
    val_files = sorted([f.name for f in VAL_IMAGES_DIR.iterdir() if f.suffix.lower() == ".jpg"])
    if len(val_files) == 0:
        raise RuntimeError(f"[ERROR] Brak plików .jpg w katalogu walidacyjnym: {VAL_IMAGES_DIR}")    
    
    results = []
    processing_times = []
    ious = []

    # Przetwarzanie pierwszych 100 obrazów
    for file_name in tqdm(val_files[:100], desc="Detekcja tablic i OCR"):
        img_path = VAL_IMAGES_DIR / file_name
        img_bgr = cv2.imread(str(img_path))
        
        start = time.time()
        
        # Detekcja tablicy za pomocą YOLO        
        detections = model(str(img_path))[0].boxes.data
        detected_bbox = None
        if detections is None or len(detections) == 0:
            ocr_plate_num = ""
        else:
            # Wybór najlepszej detekcji
            best_det = detections[np.argmax(detections[:, 4].cpu().numpy())]
            xmin, ymin, xmax, ymax = map(int, best_det[:4].cpu().numpy())
            detected_bbox = (map(int, best_det[:4].cpu().numpy()))
            
            # Przycięcie części obrazu z tablicą (z małym marginesem)
            plate_w = xmax - xmin
            new_xmin = min(max(xmin + int(0.09 * plate_w), xmin), xmax)
            crop = img_bgr[ymin:ymax, new_xmin:xmax]            
            
            # OCR na przyciętej tablicy
            ocr_plate_num = process_detected_plate(crop, file_name)
        
        processing_times.append(time.time() - start)        
        
        # Obliczenie IoU z prawdziwym bounding boxem
        iou = 0.0
        annotation = annotations.get(file_name, {})
        if annotation and detected_bbox is not None:
            annotation_bbox = annotation['bbox']
            iou = calculate_iou(detected_bbox, annotation_bbox)
        ious.append(iou)
        
        # Pobranie oczekiwanego numeru tablicy
        expected_plate_num = annotation.get('plate_number', '').upper() if annotation else ''

        results.append({
            "file_name": file_name,
            "ocr_plate_num": ocr_plate_num,
            "expected_plate_num": expected_plate_num,
            "iou": iou
        })
    
    # Obliczanie statystyk dokładności
    valid_results = [r for r in results if r["expected_plate_num"]]
    correct = sum(1 for r in valid_results if r["expected_plate_num"] in r["ocr_plate_num"])
    accuracy_percent = 100.0 * correct / len(valid_results) if valid_results else 0.0    
    
    # Obliczanie średniego IoU
    valid_ious = [iou for iou in ious if iou > 0]
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    # Obliczanie statystyk czasowych
    num_processed = len(results)
    total_time = sum(processing_times)
    avg_time = total_time / num_processed if num_processed > 0 else 0.0    
    time_for_100 = total_time * (100 / num_processed) if num_processed > 0 else float('inf')
    
    # Obliczanie końcowej oceny
    final_grade = calculate_final_grade(accuracy_percent, time_for_100)   
    
    with open(str(OCR_RESULTS), "w", encoding="utf-8") as f:
        f.write("file_name,expected_plate_num,ocr_plate_num,iou\n")
        for r in results:
            f.write(f"{r['file_name']},{r['expected_plate_num']},{r['ocr_plate_num']},{r['iou']:.4f}\n")
    
    print("\n ------------[WYNIK]------------")
    print(f"\nPrzetworzono {num_processed} obrazów.")    
    print(f"Poprawnie odczytane / zbiór walidacyjny: {correct} / {len(valid_results)}")
    print(f"Accuracy OCR: {accuracy_percent:.2f}%")    
    print(f"Łączny czas przetwarzania {num_processed} obrazów: {total_time:.2f}s")
    print(f"Śr. czas na obraz: {avg_time:.3f}s")
    print(f"Ekstrapolowany czas na 100 obrazów: {time_for_100:.2f}s")
    print(f"Średni IoU: {mean_iou:.4f}")
    print(f"Ostateczna ocena końcowa: {final_grade:.1f}")
    print("\n ------------[WYNIK]------------")

if __name__ == "__main__":
    detection_and_ocr()
