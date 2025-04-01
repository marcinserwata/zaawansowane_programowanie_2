import cv2
import numpy as np

def task1():
    # 1. Wyswietlenie pojedynczych kanalow na obrazie
    img = cv2.imread("image.jpg")
    if img is None:
        print("Error: image.jpg not found")
        return

    B, G, R = cv2.split(img)

    cv2.imshow("Original", img)
    cv2.imshow("Blue Channel", B)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Red Channel", R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("blue_channel.jpg", B)
    cv2.imwrite("green_channel.jpg", G)
    cv2.imwrite("red_channel.jpg", R)
    print("Channels saved as blue_channel.jpg, green_channel.jpg, red_channel.jpg")

def task2():
    # 2. Analiza cech uwidaczniajacych sie w poszczegolnych kanalach
    img = cv2.imread("colorful.jpg")
    if img is None:
        print("Error: colorful.jpg not found")
        return

    B, G, R = cv2.split(img)
    cv2.imshow("Original Colorful Image", img)
    cv2.imshow("Blue Channel", B)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Red Channel", R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Task2: Porownaj poszczegolne kanaly i zwroc uwage, czy jakis obiekt jest widoczny tylko na jednym z nich.")

def task3():
    # 3. Rekonstrukcja obrazu po manipulacji kanalami
    img = cv2.imread("image.jpg")
    if img is None:
        print("Error: image.jpg not found")
        return

    B, G, R = cv2.split(img)

    swapped = cv2.merge([R, B, G])
    cv2.imshow("Swapped Channels (R, B, G)", swapped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    zero_green = cv2.merge([B, np.zeros_like(G), R])
    cv2.imshow("Green Channel Zeroed", zero_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task4():
    # 4. Wzmocnienie jednego z kanalow (np. kanal czerwony)
    img = cv2.imread("image.jpg")
    if img is None:
        print("Error: image.jpg not found")
        return

    B, G, R = cv2.split(img)
    R_enhanced = cv2.add(R, 50)
    enhanced_img = cv2.merge([B, G, R_enhanced])
    cv2.imshow("Enhanced Red Channel", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task5():
    # 5. Zastosowanie maski do selektywnej modyfikacji kanalow
    img = cv2.imread("red_car.jpg")
    if img is None:
        print("Error: red_car.jpg not found")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    h, s, v = cv2.split(hsv)
    s = np.where(mask > 0, cv2.add(s, 50), s)
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv_modified = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    cv2.imshow("Original Red Car", img)
    cv2.imshow("Red Mask", mask)
    cv2.imshow("Enhanced Red Saturation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task6():
    # 6. Eksperymentowanie z logiem OpenCV
    img = cv2.imread("opencv_logo.png")
    if img is None:
        print("Error: opencv_logo.png not found")
        return

    B, G, R = cv2.split(img)
    cv2.imshow("Original Logo", img)
    cv2.imshow("Blue Channel", B)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Red Channel", R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    swapped_logo = cv2.merge([R, G, B])
    cv2.imshow("Swapped Colors Logo (B and R swapped)", swapped_logo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    removed_green = cv2.merge([B, np.zeros_like(G), R])
    cv2.imshow("Logo with Green Channel Removed", removed_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Task 1: Wyswietlenie pojedynczych kanalow")
    task1()

    print("Task 2: Analiza cech w poszczegolnych kanalach")
    task2()

    print("Task 3: Rekonstrukcja obrazu po manipulacji kanalami")
    task3()

    print("Task 4: Wzmocnienie jednego z kanalow")
    task4()

    print("Task 5: Selektywna modyfikacja kanalow z uzyciem maski")
    task5()

    print("Task 6: Eksperymenty z logiem OpenCV")
    task6()
