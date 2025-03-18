import cv2
import numpy as np

img = cv2.imread("image.png")
if img is None:
    print("Blad wczytywania obrazu. Sprawdz nazwe pliku.")
    exit()

# 1. Wybor ROI na podstawie wspolrzednych
roi1 = img[0:100, 0:100]
cv2.imshow("ROI - lewy gorny rog (100x100)", roi1)
cv2.waitKey(0)

# 2. Przyciecie dolnej polowy obrazu
(h, w) = img.shape[:2]
dolna_polowa = img[h//2:h, 0:w]
cv2.imshow("Dolna polowa obrazu", dolna_polowa)
cv2.waitKey(0)

# 3. Przyciecie prawej polowy obrazu
prawa_polowa = img[0:h, w//2:w]
cv2.imshow("Prawa polowa obrazu", prawa_polowa)
cv2.waitKey(0)

# 4. Dynamiczny wybor ROI
print("Podaj wspolrzedne ROI:")
try:
    startX = int(input("startX: "))
    endX   = int(input("endX: "))
    startY = int(input("startY: "))
    endY   = int(input("endY: "))
except ValueError:
    print("Niepoprawne wartosci. Uzycie domyslnych wartosci.")
    startX, endX, startY, endY = 0, 100, 0, 100

roi_dynamic = img[startY:endY, startX:endX]
cv2.imshow("Dynamiczny ROI", roi_dynamic)
cv2.waitKey(0)

# 5. Kadrowanie twarzy (przykladowe wspolrzedne)
face_x = 50
face_y = 50
face_w = 200
face_h = 200
if face_x + face_w <= w and face_y + face_h <= h:
    face_roi = img[face_y:face_y+face_h, face_x:face_x+face_w]
    cv2.imshow("Kadrowanie twarzy", face_roi)
    cv2.waitKey(0)
else:
    print("Wspolrzedne kadrowania twarzy wykraczaja poza rozmiar obrazu.")

# 6. Kopiowanie i wklejanie fragmentu obrazu
fragment = img[0:100, 0:100]
img_copy = img.copy()
if img_copy.shape[0] >= 250 and img_copy.shape[1] >= 250:
    img_copy[150:250, 150:250] = fragment
    cv2.imshow("Kopiowanie i wklejanie fragmentu", img_copy)
    cv2.waitKey(0)
else:
    print("Obraz zbyt maly, aby wkleic fragment w zadanym miejscu.")

# 7. Podzial obrazu na siatke 3x3
cell_h = h // 3
cell_w = w // 3
for i in range(3):
    for j in range(3):
        cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
        window_name = "Czesc {}-{}".format(i+1, j+1)
        cv2.imshow(window_name, cell)
        cv2.waitKey(0)

# 8. Animacja przesuwajacego sie ROI
roi_width = 200
roi_height = 200
if w >= roi_width and h >= roi_height:
    for x in range(0, w - roi_width + 1, 10):
        roi_anim = img[0:roi_height, x:x+roi_width]
        cv2.imshow("Animacja ROI", roi_anim)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
else:
    print("Obraz zbyt maly dla animacji ROI o wymiarach {}x{}.".format(roi_width, roi_height))

# 9. Zapis przycietego obrazu
if h >= 300 and w >= 300:
    cropped = img[0:300, 0:300]
    cv2.imwrite("cropped_image.jpg", cropped)
    cv2.imshow("Przyciety obraz (300x300)", cropped)
    cv2.waitKey(0)
else:
    print("Obraz zbyt maly do przyciecia do wymiarow 300x300.")

cv2.destroyAllWindows()
