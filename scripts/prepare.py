import os
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.xml")

YOLO_DIR = os.path.join(ROOT_DIR, "yolo_data")
IMAGES_TRAIN_DIR = os.path.join(YOLO_DIR, "images", "train")
IMAGES_VAL_DIR = os.path.join(YOLO_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(YOLO_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(YOLO_DIR, "labels", "val")

for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

tree = ET.parse(ANNOTATIONS_FILE)
root = tree.getroot()

# Mapa: nazwa_pliku.jpg → lista bboxów [(xmin, ymin, xmax, ymax, class_label, plate_number), ...]
annotations = {}
for image in root.findall("image"):
    img_name = image.attrib["name"]
    width = float(image.attrib["width"])
    height = float(image.attrib["height"])

    bboxes = []
    for box in image.findall("box"):
        label = box.attrib["label"]  # powinno być "plate"
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        # Pobranie numeru tablicy (atrybut "plate number")
        plate_attr = box.find("attribute[@name='plate number']")
        plate_number = plate_attr.text if plate_attr is not None else ""

        bboxes.append((xtl, ytl, xbr, ybr, label, plate_number))
    annotations[img_name] = {"size": (width, height), "bboxes": bboxes}

# 2. Podział nazw plików na train/val (70%/30%)
wszystkie_pliki = list(annotations.keys())
train_files, val_files = train_test_split(
    wszystkie_pliki, test_size=0.3, random_state=42
)

# 3. Funkcja konwersji bbox → format YOLO (klasa 0 dla tablicy)
def convert_to_yolo(bbox, img_width, img_height):
    xtl, ytl, xbr, ybr, label, plate_num = bbox
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    box_w = xbr - xtl
    box_h = ybr - ytl
    # Normalizacja
    x_center /= img_width
    y_center /= img_height
    box_w /= img_width
    box_h /= img_height
    # Klasa "plate" ustawiamy jako 0
    return 0, x_center, y_center, box_w, box_h

# 4. Generowanie plików obrazów + labeli
for subset, file_list, img_dest, label_dest in [
    ("train", train_files, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR),
    ("val", val_files, IMAGES_VAL_DIR, LABELS_VAL_DIR),
]:
    for fname in file_list:
        # Kopiujemy obraz do odpowiedniego folderu
        src_img_path = os.path.join(IMAGES_DIR, fname)
        dst_img_path = os.path.join(img_dest, fname)
        shutil.copy(src_img_path, dst_img_path)

        # Otwieramy info o obrazku
        w, h = annotations[fname]["size"]
        bboxes = annotations[fname]["bboxes"]

        # Tworzymy plik .txt z adnotacjami YOLO
        label_file = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_dest, label_file)
        with open(label_path, "w") as f:
            for bbox in bboxes:
                class_id, x_c, y_c, bw, bh = convert_to_yolo(bbox, w, h)
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

print("Przygotowano dane w formacie YOLO w katalogu yolo_data/.")
