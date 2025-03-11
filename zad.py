import cv2
import numpy as np

image = cv2.imread('img.png')
if image is None:
    print("Nie udało się wczytać obrazu! Sprawdź ścieżkę.")
    exit()

height, width = image.shape[:2]
print(f"Rozmiar obrazu: szerokość={width}, wysokość={height}")

# 1. Odczyt wartości piksela (lewygórny róg)
pixel = image[0, 0]
B, G, R = pixel
print(f"1. Piksel w lewym górnym rogu (0,0): R={R}, G={G}, B={B}")

# 2. Modyfikacja pojedynczego piksela (prawy dolny róg)
image_before = image.copy()

image[height - 1, width - 1] = [0, 0, 255]
print("2. Zmieniono kolor piksela w prawym dolnym rogu na czerwony.")

cv2.imshow("Obraz przed zmianą", image_before)
cv2.imshow("Obraz po zmianie (prawy dolny piksel czerwony)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Znajdowanie środka obrazu
center_x = width // 2
center_y = height // 2
center_pixel = image[center_y, center_x]
B, G, R = center_pixel
print(f"3. Piksel w środku obrazu ({center_x}, {center_y}): R={R}, G={G}, B={B}")

# 4. Zamiana wartości piksela na czarny (użytkownik podaje współrzędne)
user_input = input("4. Podaj współrzędne piksela (x,y): ")
try:
    x_str, y_str = user_input.split(',')
    x = int(x_str.strip())
    y = int(y_str.strip())
    if x < 0 or x >= width or y < 0 or y >= height:
        print("Podane współrzędne wykraczają poza wymiary obrazu!")
    else:
        image[y, x] = [0, 0, 0]  # czarny w BGR
        print(f"Ustawiono piksel na ({x}, {y}) na czarny.")
except Exception as e:
    print("Błąd przy przetwarzaniu współrzędnych:", e)

cv2.imshow("Obraz po ustawieniu piksela na czarny", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Kolorowanie fragmentu obrazu (lewygórna ćwiartka na niebiesko)
image_quadrant = image.copy()
half_width = width // 2
half_height = height // 2
image_quadrant[0:half_height, 0:half_width] = [255, 0, 0]
cv2.imshow("5. Obraz z lewą górną ćwiartką niebieską", image_quadrant)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Wypełnienie obszaru – kwadrat 100x100 z centralnym punktem
image_square = image.copy()
square_size = 100
top_left_x = center_x - square_size // 2
top_left_y = center_y - square_size // 2
image_square[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = [0, 0, 255]
cv2.imshow("6. Obraz z czerwonym kwadratem 100x100 w centrum", image_square)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Wycięcie fragmentu obrazu (dzielenie na 9 części, wycięcie centralnego fragmentu)
cell_width = width // 3
cell_height = height // 3
central_fragment = image[cell_height:2 * cell_height, cell_width:2 * cell_width]
cv2.imshow("7. Centralny fragment obrazu", central_fragment)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Modyfikacja całego wiersza pikseli (100. wiersz na zielony)
image_row_mod = image.copy()
if height > 100:
    image_row_mod[100, :] = [0, 255, 0]
    print("8. Zmieniono kolor 100. wiersza na zielony.")
    cv2.imshow("Obraz z 100. wierszem zielonym", image_row_mod)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Obraz ma za mało wierszy (mniej niż 101).")

# 9. Zmiana wartości pikseli w zakresie (50,50) do (100,100) na biały
image_area = image.copy()
cv2.imshow("9. Przed zmianą (obszar)", image_area)
cv2.waitKey(0)
image_area[50:101, 50:101] = [255, 255, 255]
cv2.imshow("9. Po zmianie (obszar biały)", image_area)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 10. Sprawdzenie różnicy wartości pikseli (np. (50,50) i (200,200))
if height > 200 and width > 200:
    pixel1 = image[50, 50]
    pixel2 = image[200, 200]
    diff = np.abs(pixel1.astype(int) - pixel2.astype(int))
    print(f"10. Różnice między pikselem (50,50) a (200,200): B={diff[0]}, G={diff[1]}, R={diff[2]}")
else:
    print("Obraz jest zbyt mały, aby porównać piksele w (50,50) i (200,200).")

# 11. Znajdowanie najjaśniejszego piksela w obrazie
brightness = image.sum(axis=2)
max_index = np.unravel_index(np.argmax(brightness, axis=None), brightness.shape)
max_pixel = image[max_index]
print(f"11. Najjaśniejszy piksel znajduje się w {max_index} o wartości (B, G, R): {max_pixel}")

cv2.waitKey(0)
cv2.destroyAllWindows()
