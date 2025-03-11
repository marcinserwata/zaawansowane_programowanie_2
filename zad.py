import cv2
import imutils

def load_image(path="image.jpg"):
    image = cv2.imread(path)
    if image is None:
        print("Nie znaleziono obrazu! Upewnij sie, ze plik istnieje.")
        exit()
    return image

def rotate_image_center(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def main():
    image = load_image("image.jpg")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 1. Obrot o 45 stopni wokol srodka
    rotated_45 = rotate_image_center(image, 45)
    cv2.imshow("Oryginalny obraz", image)
    cv2.imshow("Obrot 45 stopni", rotated_45)
    cv2.waitKey(0)

    # 2. Obrot o -90 stopni wokol srodka
    rotated_minus90 = rotate_image_center(image, -90)
    cv2.imshow("Obrot -90 stopni", rotated_minus90)
    cv2.waitKey(0)

    # 3. Obrot o 30 stopni wzgledem lewego gornego naroznika (0,0)
    M_corner = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
    rotated_corner = cv2.warpAffine(image, M_corner, (w, h))
    cv2.imshow("Obrot 30 stopni wokol (0,0)", rotated_corner)
    cv2.waitKey(0)

    # 4. Obrot o dowolny kat pobrany od uzytkownika (wokol srodka)
    try:
        angle_input = float(input("Podaj kat obrotu: "))
    except ValueError:
        print("Nieprawidlowa wartosc kata.")
        return
    rotated_custom = rotate_image_center(image, angle_input)
    cv2.imshow("Obrot o kat podany przez uzytkownika", rotated_custom)
    cv2.waitKey(0)

    # 5. Obrot o 180 stopni za pomoca imutils.rotate
    rotated_180_imutils = imutils.rotate(image, 180)
    cv2.imshow("Obrot 180 stopni (imutils.rotate)", rotated_180_imutils)
    cv2.waitKey(0)

    # 6. Obrot bez przycinania (rotate_bound) o -33 stopni
    rotated_bound = imutils.rotate_bound(image, -33)
    cv2.imshow("Obrot -33 stopni (rotate_bound)", rotated_bound)
    cv2.waitKey(0)

    # 7. Porownanie warpAffine i imutils.rotate dla obrotu o 60 stopni
    rotated_warpAffine = rotate_image_center(image, 60)
    rotated_imutils = imutils.rotate(image, 60)
    cv2.imshow("Obrot 60 stopni (warpAffine)", rotated_warpAffine)
    cv2.imshow("Obrot 60 stopni (imutils.rotate)", rotated_imutils)
    cv2.waitKey(0)

    # 8. Obrot sekwencyjny: trzy obroty po 30 stopni
    rotated_seq = image.copy()
    for i in range(3):
        rotated_seq = rotate_image_center(rotated_seq, 30)
    rotated_90 = rotate_image_center(image, 90)
    cv2.imshow("Sekwencyjny obrot 3 x 30 stopni", rotated_seq)
    cv2.imshow("Pojedynczy obrot 90 stopni", rotated_90)
    cv2.waitKey(0)

    # 9. Obrot o 75 stopni i zapis do pliku
    rotated_75 = rotate_image_center(image, 75)
    cv2.imwrite("rotated_output.jpg", rotated_75)
    cv2.imshow("Obrot 75 stopni", rotated_75)
    cv2.waitKey(0)

    # 10. Obrot w petli: obracamy obraz co 15 stopni od 0 do 360
    for angle in range(0, 361, 15):
        rotated_loop = rotate_image_center(image, angle)
        cv2.imshow("Obrot w petli", rotated_loop)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
