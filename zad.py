import cv2

# 1. Wczytaj i wyświetl obraz z podanej ścieżki.
sciezka = 'image.jpg'

img = cv2.imread(sciezka)

if img is None:
    print("Błąd: Nie można wczytać obrazu. Sprawdź podaną ścieżkę:", sciezka)
else:
    cv2.imshow("Obraz - oryginalny", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2. Wczytaj zdjęcie w kolorze i wyświetl liczbę kanałów.
img_color = cv2.imread(sciezka, cv2.IMREAD_COLOR)
if img_color is None:
    print("Błąd: Nie można wczytać obrazu w kolorze.")
else:
    liczba_kanalow = img_color.shape[2]
    print("Liczba kanałów w obrazie kolorowym:", liczba_kanalow)

# 3. Wczytaj zdjęcie w odcieniach szarości i wyświetl liczbę kanałów.
img_gray = cv2.imread(sciezka, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    print("Błąd: Nie można wczytać obrazu w odcieniach szarości.")
else:
    print("Liczba kanałów w obrazie szarym: 1 (wymiary tablicy:", img_gray.shape,")")


# 4. Wczytaj obraz w skali szarości i zapisz go jako nowy plik.
nazwa_pliku = "gray_image.jpg"
zapis = cv2.imwrite(nazwa_pliku, img_gray)
if zapis:
    print("Obraz w skali szarości zapisano jako:", nazwa_pliku)
else:
    print("Błąd podczas zapisywania obrazu.")


# 5. Otwórz dwa obrazy jednocześnie w osobnych oknach. Upewnij się, że można je zamknąć niezależnie.
img1 = cv2.imread(sciezka)
img2 = cv2.imread(sciezka)
if img1 is None or img2 is None:
    print("Błąd: Nie można wczytać jednego z obrazów.")
else:
    cv2.imshow("Okno 1", img1)
    cv2.imshow("Okno 2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 6. Zmień rozmiar okna wyświetlania obrazu (cv2.WINDOW_NORMAL).
cv2.namedWindow("Okno zmieniane", cv2.WINDOW_NORMAL)
cv2.imshow("Okno zmieniane", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
