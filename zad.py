import cv2
import numpy as np

# 1. Porownanie metod dodawania
img = cv2.imread("image.jpg")
if img is None:
    print("Blad wczytywania obrazu. Sprawdz nazwe pliku.")
    exit()

img_np_bright = img + 50

img_cv_bright = cv2.add(img, 50)

cv2.imshow("Oryginal", img)
cv2.imshow("Jasnosc +50 (NumPy)", img_np_bright)
cv2.imshow("Jasnosc +50 (cv2.add)", img_cv_bright)
cv2.waitKey(0)

# 2. Symulacja efektu "przepalenia" obrazu

img_np_burn = img + 150
img_cv_burn = cv2.add(img, 150)

cv2.imshow("Przepalony (NumPy)", img_np_burn)
cv2.imshow("Przepalony (cv2.add)", img_cv_burn)
cv2.waitKey(0)

# 3. Przyciemnianie obrazu

img_np_dark = img - 80
img_cv_dark = cv2.subtract(img, 80)

cv2.imshow("Przyciemniony (NumPy)", img_np_dark)
cv2.imshow("Przyciemniony (cv2.subtract)", img_cv_dark)
cv2.waitKey(0)

# 4. Tworzenie wlasnego "filtra Instagram"
img_filter = img.copy()
B, G, R = cv2.split(img_filter)
B = cv2.add(B, 10)      # Niebieski
G = cv2.subtract(G, 20) # Zielony
R = cv2.add(R, 30)      # Czerwony
img_filter = cv2.merge((B, G, R))

cv2.imshow("Filtr Instagram", img_filter)
cv2.waitKey(0)

# 5. Zastosowanie arytmetyki do detekcji zmian w obrazach
img1 = cv2.imread("input1.jpg")
img2 = cv2.imread("input2.jpg")
if img1 is None or img2 is None:
    print("Blad wczytywania obrazow. Sprawdz nazwy plikow: input1.jpg oraz input2.jpg.")
    exit()

if img1.shape != img2.shape:
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

diff = cv2.absdiff(img1, img2)

cv2.imshow("Obraz 1", img1)
cv2.imshow("Obraz 2", img2)
cv2.imshow("Roznica (absdiff)", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Interpretacja:
# Obraz "diff" pokazuje roznice miedzy obrazami. Jasne obszary wskazuja na miejsca,
# w ktorych wystapily zmiany, np. przesuniety obiekt lub niewielkie roznice w scenie.
