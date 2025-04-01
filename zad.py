import cv2
import numpy as np

# 1. Maskowanie obszaru twarzy
def mask_face():
    img = cv2.imread("person.png")
    if img is None:
        print("Error")
        return

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    rows, cols = mask.shape

    center = (cols // 2, rows // 2)
    axes = (cols // 4, rows // 3)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Oryginal", img)
    cv2.imshow("Maska", mask)
    cv2.imshow("Maska na twarzy", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Ukrywanie okreslonego obszaru twarzy (oczy)
def hide_eyes():
    img = cv2.imread("person.png")
    if img is None:
        print("Error")
        return

    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    rows, cols = mask.shape

    x1 = 250
    y1 = 180
    x2 = 500
    y2 = 230
    cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Oryginal", img)
    cv2.imshow("Maska na oczach", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3. Wykorzystanie maski do ekstrakcji koloru
def extract_color():
    img = cv2.imread("bird.jpg")
    if img is None:
        print("Error")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Oryginal", img)
    cv2.imshow("Maska koloru", mask)
    cv2.imshow("Sam kolor", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Maskowanie obszaru twarzy")
    mask_face()
    print("2. Ukrywanie okreslonego obszaru twarzy (oczy)")
    hide_eyes()
    print("3. Wykorzystanie maski do ekstrakcji koloru")
    extract_color()
