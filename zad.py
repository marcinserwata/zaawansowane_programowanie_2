import cv2
import numpy as np

# 1. Rysowanie linii
image1 = np.zeros((500, 500, 3), dtype=np.uint8)
center = (image1.shape[1] // 2, image1.shape[0] // 2)
bottom_right = (image1.shape[1] - 1, image1.shape[0] - 1)
cv2.line(image1, center, bottom_right, (255, 0, 0), 2)
cv2.imshow('1: Rysowanie linii', image1)
cv2.waitKey(0)

# 2. Rysowanie prostokątów
image2 = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(image2, (0, 0), (100, 50), (0, 255, 0), -1)
cv2.rectangle(image2, (400 - 100, 400 - 50), (400, 400), (0, 0, 255), 3)
cv2.imshow('2: Rysowanie prostokątów', image2)
cv2.waitKey(0)

# 3. Rysowanie okręgów
image3 = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.circle(image3, (40, 40), 40, (255, 0, 0), -1)
cv2.circle(image3, (150, 150), 60, (0, 0, 255), -1)
cv2.imshow('3: Rysowanie okręgów', image3)
cv2.waitKey(0)

# 4. Złożona figura
image4 = np.zeros((400, 400, 3), dtype=np.uint8)
center_img = (image4.shape[1] // 2, image4.shape[0] // 2)
half_square = 50
top_left = (center_img[0] - half_square, center_img[1] - half_square)
bottom_right = (center_img[0] + half_square, center_img[1] + half_square)
cv2.rectangle(image4, top_left, bottom_right, (255, 255, 255), 2)
cv2.circle(image4, center_img, 30, (0, 0, 255), -1)
cv2.imshow('4: Złożona figura', image4)
cv2.waitKey(0)

# 5. Eksperymentowanie z pętlą
image5 = np.zeros((300, 300, 3), dtype=np.uint8)
center_img = (150, 150)
initial_size = 20
num_squares = 5

for i in range(num_squares):
    size = initial_size + i * 20
    half_size = size // 2
    top_left = (center_img[0] - half_size, center_img[1] - half_size)
    bottom_right = (center_img[0] + half_size, center_img[1] + half_size)
    cv2.rectangle(image5, top_left, bottom_right, (255, 255, 255), 2)

cv2.imshow('5: Eksperymentowanie z pętlą', image5)
cv2.waitKey(0)

# 6. Zamazywanie (zasłanianie) szczegółów na zdjęciu
image_path = 'profile.png'
image6 = cv2.imread(image_path)
if image6 is None:
    print("Nie znaleziono pliku", image_path)
else:
    left_eye = (
        int(image6.shape[1] * 0.51),
        int(image6.shape[0] * 0.37)
    )
    right_eye = (
        int(image6.shape[1] * 0.64),
        int(image6.shape[0] * 0.36)
    )
    eye_radius = 15
    cv2.circle(image6, left_eye, eye_radius, (0, 0, 255), -1)
    cv2.circle(image6, right_eye, eye_radius, (0, 0, 255), -1)

    mouth_top_left = (
        int(image6.shape[1] * 0.5),
        int(image6.shape[0] * 0.5)
    )
    mouth_bottom_right = (
        int(image6.shape[1] * 0.65),
        int(image6.shape[0] * 0.60)
    )
    cv2.rectangle(image6, mouth_top_left, mouth_bottom_right, (0, 255, 0), -1)

    face_center = (int(image6.shape[1] * 0.53), int(image6.shape[0] * 0.37))
    face_radius = int(min(image6.shape[0], image6.shape[1]) * 0.35)
    cv2.circle(image6, face_center, face_radius, (255, 0, 0), 2)

    cv2.imshow('6: Zamazywanie szczegółów na zdjęciu', image6)
    cv2.waitKey(0)

cv2.destroyAllWindows()