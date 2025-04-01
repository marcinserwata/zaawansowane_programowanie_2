import cv2
import numpy as np


def draw_triangle(img, center, size):
    pts = np.array([[center,
                     (center[0] - size, center[1] + size),
                     (center[0] + size, center[1] + size)]], dtype=np.int32)
    cv2.fillPoly(img, pts, 255)


def draw_circle(img, center, radius):
    cv2.circle(img, center, radius, 255, -1)


# 1. Kombinacja roznych kształtow i operacji bitowych

img_triangle = np.zeros((400, 400), dtype=np.uint8)
img_circle = np.zeros((400, 400), dtype=np.uint8)

center = (200, 150)
size = 100
draw_triangle(img_triangle, center, size)
draw_circle(img_circle, center, size)

img_and = cv2.bitwise_and(img_triangle, img_circle)
img_or = cv2.bitwise_or(img_triangle, img_circle)
img_xor = cv2.bitwise_xor(img_triangle, img_circle)
img_not_triangle = cv2.bitwise_not(img_triangle)
img_not_circle = cv2.bitwise_not(img_circle)

cv2.imshow("Trojkat", img_triangle)
cv2.imshow("Okrag", img_circle)
cv2.imshow("AND", img_and)
cv2.imshow("OR", img_or)
cv2.imshow("XOR", img_xor)
cv2.imshow("NOT Trojkat", img_not_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_triangle_shifted = np.zeros((400, 400), dtype=np.uint8)
img_circle_shifted = np.zeros((400, 400), dtype=np.uint8)

center_shifted = (250, 250)
draw_triangle(img_triangle_shifted, center_shifted, size)
draw_circle(img_circle_shifted, center_shifted, size)

img_and_shifted = cv2.bitwise_and(img_triangle_shifted, img_circle_shifted)
img_or_shifted = cv2.bitwise_or(img_triangle_shifted, img_circle_shifted)
img_xor_shifted = cv2.bitwise_xor(img_triangle_shifted, img_circle_shifted)

cv2.imshow("Przesuniety Trojkat", img_triangle_shifted)
cv2.imshow("Przesuniety Okrag", img_circle_shifted)
cv2.imshow("Przesuniety AND", img_and_shifted)
cv2.imshow("Przesuniety OR", img_or_shifted)
cv2.imshow("Przesuniety XOR", img_xor_shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Zastosowanie operacji XOR do wykrywania różnic między obrazami

img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Blad: Jeden lub oba obrazy nie zostaly znalezione. Upewnij siw, ze 'image1.png' oraz 'image2.png' istnieja.")
else:
    img_diff = cv2.bitwise_xor(img1, img2)

    cv2.imshow("Roznice (XOR)", img_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
