import cv2
import numpy as np
import imutils

img = cv2.imread("image.jpg")
if img is None:
    print("Nie znaleziono obrazu. Upewnij sie, że plik 'image.jpg' istnieje.")
    exit()

# 1. Podstawowe przesunięcie
cv2.imshow("Oryginalny", img)
cv2.waitKey(0)
M = np.float32([[1, 0, 30], [0, 1, 40]])
translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Przesuniecie 30 w prawo, 40 w dol", translated)
cv2.waitKey(0)

# 2. Przesunięcie w przeciwnym kierunku
M2 = np.float32([[1, 0, -20], [0, 1, -50]])
translated2 = cv2.warpAffine(img, M2, (img.shape[1], img.shape[0]))
cv2.imshow("Przesuniecie 20 w lewo, 50 w gore", translated2)
cv2.waitKey(0)

# 3. Eksperymentowanie z dużymi wartościami przesunięcia
tx = img.shape[1] // 2 + 10
ty = img.shape[0] // 2 + 10
M3 = np.float32([[1, 0, tx], [0, 1, ty]])
translated3 = cv2.warpAffine(img, M3, (img.shape[1], img.shape[0]))
cv2.imshow("Duze przesuniecie", translated3)
cv2.waitKey(0)

# 4. Wykorzystanie funkcji imutils.translate
translated_imutils = imutils.translate(img, 100, 50)
cv2.imshow("imutils.translate: 100 w prawo, 50 w dol", translated_imutils)
cv2.waitKey(0)

# 5. Dynamiczne przesunięcie na podstawie parametrów użytkownika
try:
    tx_input = int(input("Podaj wartosc przesuniecia w poziomie (tx): "))
    ty_input = int(input("Podaj wartosc przesuniecia w pionie (ty): "))
except ValueError:
    print("Podana wartosc nie jest liczba całkowita.")
    exit()

M_dynamic = np.float32([[1, 0, tx_input], [0, 1, ty_input]])
translated_dynamic = cv2.warpAffine(img, M_dynamic, (img.shape[1], img.shape[0]))
cv2.imshow("Dynamiczne przesuniecie", translated_dynamic)
cv2.waitKey(0)

cv2.destroyAllWindows()
