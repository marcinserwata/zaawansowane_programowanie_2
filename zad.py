import cv2
import numpy as np

img = cv2.imread("image.jpg")
if img is None:
    print("Blad wczytywania obrazu. Sprawdz nazwe pliku.")
    exit()

# 1. Odbicie poziome
img_flip_horizontal = cv2.flip(img, 1)
cv2.imshow("Odbicie poziome", img_flip_horizontal)
cv2.waitKey(0)

# 2. Odbicie pionowe
img_flip_vertical = cv2.flip(img, 0)
cv2.imshow("Odbicie pionowe", img_flip_vertical)
cv2.waitKey(0)

# 3. Odbicie wzgledem obu osi
img_flip_both = cv2.flip(img, -1)
cv2.imshow("Odbicie obu osi", img_flip_both)
cv2.waitKey(0)

# 4. Porownanie efektow
top_row = np.hstack((img, img_flip_horizontal))
bottom_row = np.hstack((img_flip_vertical, img_flip_both))
comparison = np.vstack((top_row, bottom_row))
cv2.imshow("Porownanie efektow", comparison)
cv2.waitKey(0)

# 5. Zastosowanie odbicia na wybranym obszarze
(h, w) = img.shape[:2]
x1 = w // 4
x2 = 3 * w // 4
y1 = h // 4
y2 = 3 * h // 4
roi = img[y1:y2, x1:x2]
roi_flip = cv2.flip(roi, 1)
img_modified = img.copy()
img_modified[y1:y2, x1:x2] = roi_flip
cv2.imshow("Odbicie na wybranym obszarze", img_modified)
cv2.waitKey(0)

# 6. Odbicie na podstawie wyboru uzytkownika
choice = input("Podaj sposob odbicia (0 - pionowe, 1 - poziome, -1 - oba): ")
try:
    choice = int(choice)
    if choice not in [0, 1, -1]:
        print("Niepoprawna wartosc. Uzycie odbicia poziomego jako domyslnego.")
        choice = 1
except ValueError:
    print("Niepoprawna wartosc. Uzycie odbicia poziomego jako domyslnego.")
    choice = 1

img_user_flip = cv2.flip(img, choice)
cv2.imshow("Odbicie wg wyboru uzytkownika", img_user_flip)
cv2.waitKey(0)
cv2.destroyAllWindows()
