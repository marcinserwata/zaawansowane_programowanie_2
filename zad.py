import cv2
import imutils

img = cv2.imread("image.jpg")
if img is None:
    print("Blad wczytywania obrazu. Sprawdz nazwe pliku.")
    exit()

# 1. Zmniejszenie obrazu o połowę
width_half = int(img.shape[1] * 0.5)
height_half = int(img.shape[0] * 0.5)
dim_half = (width_half, height_half)
img_half = cv2.resize(img, dim_half, interpolation=cv2.INTER_AREA)
cv2.imshow("Obraz zmniejszony o 50%", img_half)
cv2.waitKey(0)

# 2. Powiększenie obrazu dwukrotnie
width_double = int(img.shape[1] * 2)
height_double = int(img.shape[0] * 2)
dim_double = (width_double, height_double)
img_double = cv2.resize(img, dim_double, interpolation=cv2.INTER_LINEAR)
cv2.imshow("Obraz powiekszony 2x (INTER_LINEAR)", img_double)
cv2.waitKey(0)

# 3. Zmiana rozmiaru na konkretną wartość
img_specific = cv2.resize(img, (200, 300))
cv2.imshow("Obraz 200x300", img_specific)
cv2.waitKey(0)

# 4. Porównanie różnych metod interpolacji
methods = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4
}

for method_name, method in methods.items():
    new_width = int(img.shape[1] * 3)
    new_height = int(img.shape[0] * 3)
    resized = cv2.resize(img, (new_width, new_height), interpolation=method)
    cv2.imshow(f"Powiekszony 3x - {method_name}", resized)
    cv2.waitKey(0)

# 5. Automatyczne skalowanie na podstawie szerokości
img_width_500 = imutils.resize(img, width=500)
cv2.imshow("Skalowanie: szerokosc 500px", img_width_500)
cv2.waitKey(0)

# 6. Automatyczne skalowanie na podstawie wysokości
img_height_400 = imutils.resize(img, height=400)
cv2.imshow("Skalowanie: wysokosc 400px", img_height_400)
cv2.waitKey(0)

# 7. Efekty przy skalowaniu w dół
new_width_down = int(img.shape[1] / 5)
new_height_down = int(img.shape[0] / 5)
img_down = cv2.resize(img, (new_width_down, new_height_down), interpolation=cv2.INTER_AREA)
cv2.imshow("Zmniejszony obraz 5x (INTER_AREA)", img_down)
cv2.waitKey(0)

# 8. Efekty przy skalowaniu w górę
new_width_up = int(img.shape[1] * 4)
new_height_up = int(img.shape[0] * 4)
img_up_cubic = cv2.resize(img, (new_width_up, new_height_up), interpolation=cv2.INTER_CUBIC)
img_up_lanczos = cv2.resize(img, (new_width_up, new_height_up), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("Powiekszony 4x (INTER_CUBIC)", img_up_cubic)
cv2.waitKey(0)
cv2.imshow("Powiekszony 4x (INTER_LANCZOS4)", img_up_lanczos)
cv2.waitKey(0)

# 9. Dynamiczna zmiana rozmiaru w pętli
for scale in range(100, 320, 20):
    width_dyn = int(img.shape[1] * scale / 100)
    height_dyn = int(img.shape[0] * scale / 100)
    img_dyn = cv2.resize(img, (width_dyn, height_dyn), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Dynamiczna zmiana rozmiaru", img_dyn)
    cv2.waitKey(500)

# 10. Zmiana rozmiaru i zapis pliku
img_800 = imutils.resize(img, width=800)
cv2.imwrite("resized_output.jpg", img_800)
cv2.imshow("Obraz o szerokosci 800px", img_800)
cv2.waitKey(0)

cv2.destroyAllWindows()
