import cv2
import numpy as np
import os
import shutil

IMAGE_PATH = 'kostka_brukowa.png'
TARGET_WIDTH = 300
OUTPUT_DIR = 'wyciete_kostki'

SELECTED_THRESHOLD = 140
THRESHOLD_TYPE = cv2.THRESH_BINARY

MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 5000


def display_image(title, image, wait=True):
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)

if not os.path.exists(IMAGE_PATH):
    print(f"BŁĄD: Plik obrazu '{IMAGE_PATH}' nie został znaleziony.")
    exit()

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"Utworzono katalog '{OUTPUT_DIR}' na wycięte kostki.")

original_image_color_loaded = cv2.imread(IMAGE_PATH)
if original_image_color_loaded is None:
    print(f"BŁĄD: Nie można wczytać obrazu '{IMAGE_PATH}'.")
    exit()

# Zadanie 1. Klasyczne progowanie
# a. Wczytaj obraz z kostką brukową, przeskaluj go do szerokości 300 px i
#    zastosuj progowanie klasyczne (cv2.threshold) dla różnych wartości
#    progowania (np. 100, 140, 180).
original_height, original_width = original_image_color_loaded.shape[:2]
aspect_ratio = original_height / original_width
target_height = int(TARGET_WIDTH * aspect_ratio)

scaled_image_color = cv2.resize(original_image_color_loaded, (TARGET_WIDTH, target_height))
gray_image = cv2.cvtColor(scaled_image_color, cv2.COLOR_BGR2GRAY)

print(f"Obraz przeskalowany do: {TARGET_WIDTH}x{target_height} px")
display_image("Oryginalny przeskalowany", scaled_image_color, wait=False)
display_image("Obraz w skali szarosci", gray_image, wait=False)

thresholds_to_test_from_task = [100, 140, 180]
for thresh_val_task in thresholds_to_test_from_task:
    _, binary_image_task_test = cv2.threshold(gray_image, thresh_val_task, 255, THRESHOLD_TYPE)
    display_image(f"Test progow. (zad.1a): {thresh_val_task}, typ: {THRESHOLD_TYPE}", binary_image_task_test, wait=False)

# b. Zaobserwuj, jak zmienia się jakość segmentacji kostek. Która wartość
#    progowania najlepiej rozdziela kostki od tła?
# Wniosek/Obserwacja (Zadanie 1b):
#   Jakość segmentacji zależy od wybranego progu.
#   - Zbyt niski próg może powodować łączenie się kostek z jaśniejszymi elementami tła lub powstawanie "dziur" w kostkach.
#   - Zbyt wysoki próg może powodować utratę części kostek lub ich fragmentację, jeśli są ciemniejsze.
#   Należy wybrać próg, który najlepiej oddziela całe kostki (jako białe obiekty) od czarnego tła.
#   Wartość SELECTED_THRESHOLD = 140 i THRESHOLD_TYPE = cv2.THRESH_BINARY wydaje się dobrym punktem wyjścia
#   dla obrazu podobnego do tego z dokumentu. Użytkownik powinien zweryfikować to wizualnie.
print(f"\nZad.1b: Zaobserwowano wpływ progów. Do dalszej pracy wybrano próg: {SELECTED_THRESHOLD}, typ: {THRESHOLD_TYPE}")
print("Naciśnij dowolny klawisz, aby zamknąć okna z testami progowania i kontynuować...")
cv2.waitKey(0)
cv2.destroyAllWindows()

_, best_binary_image = cv2.threshold(gray_image, SELECTED_THRESHOLD, 255, THRESHOLD_TYPE)
display_image("Docelowy obraz binarny (best_binary_image)", best_binary_image, wait=False)

# Zadanie 2. Eksperymentuj z metodą cv2.findContours
# a. Na progowanym obrazie znajdź kontury przy użyciu funkcji cv2.findContours.
#    Narysuj wszystkie wykryte kontury na oryginalnym obrazie w kolorze
#    czerwonym o grubości 2px.
# b. Zmieniaj tryby (parametr mode w funkcji findContours), przetestuj
#    cv2.RETR_EXTERNAL, cv2.RETR_TREE i cv2.RETR_LIST i opisz różnice w komentarzu.

contours_display_external_task = scaled_image_color.copy()
contours_display_tree_task = scaled_image_color.copy()
contours_display_list_task = scaled_image_color.copy()

# Test cv2.RETR_EXTERNAL (Zadanie 2b)
contours_ext_task, _ = cv2.findContours(best_binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_display_external_task, contours_ext_task, -1, (0, 0, 255), 2) # Czerwone, grubość 2px (Zadanie 2a)
display_image("Zad.2: Kontury - RETR_EXTERNAL", contours_display_external_task, wait=False)
print(f"- Zad.2b (RETR_EXTERNAL): Znalazł {len(contours_ext_task)} konturów.")

# Test cv2.RETR_TREE (Zadanie 2b)
contours_tree_task, hierarchy_tree_task = cv2.findContours(best_binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_display_tree_task, contours_tree_task, -1, (0, 0, 255), 2) # Czerwone, grubość 2px
display_image("Zad.2: Kontury - RETR_TREE", contours_display_tree_task, wait=False)
print(f"- Zad.2b (RETR_TREE): Znalazł {len(contours_tree_task)} konturów.")

# Test cv2.RETR_LIST (Zadanie 2b)
contours_list_task, _ = cv2.findContours(best_binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_display_list_task, contours_list_task, -1, (0, 0, 255), 2) # Czerwone, grubość 2px
display_image("Zad.2: Kontury - RETR_LIST", contours_display_list_task, wait=False)
print(f"- Zad.2b (RETR_LIST): Znalazł {len(contours_list_task)} konturów.")

# Wnioski/Obserwacje (Zadanie 2b):
#   - cv2.RETR_EXTERNAL: Pobiera tylko skrajne, zewnętrzne kontury. Idealny do oddzielania nie nakładających się obiektów, jak kostki.
#   - cv2.RETR_TREE: Rekonstruuje pełną hierarchię zagnieżdżonych konturów. Przydatny, gdy obiekty mogą zawierać inne obiekty (np. otwory).
#                    `hierarchy_tree_task` zawiera informacje o relacjach między konturami.
#   - cv2.RETR_LIST: Pobiera wszystkie kontury, ale nie tworzy informacji o hierarchii. Wszystkie kontury są na tym samym poziomie.

print("\nNaciśnij dowolny klawisz, aby zamknąć okna z trybami konturów i kontynuować...")
cv2.waitKey(0)
cv2.destroyAllWindows()

final_contours, _ = cv2.findContours(best_binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Do dalszej pracy wybrano cv2.RETR_EXTERNAL. Liczba konturów: {len(final_contours)}")

if not final_contours:
    print("KRYTYCZNY BŁĄD: Nie znaleziono konturów. Sprawdź ustawienia progowania.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Zadanie 3. Eksperymentuj z rozdzielczością obrazu
# a. Zmieniaj rozmiar obrazu wejściowego przed detekcją konturów. Sprawdź,
#    jak zmiana rozdzielczości wpływa na liczbę i jakość wykrytych konturów.
# b. Czy zmniejszenie obrazu może poprawić detekcję?
# Wnioski/Obserwacje (Zadanie 3):
#   Zmiana rozdzielczości (TARGET_WIDTH) wpływa na:
#   - Jakość progowania: Przy mniejszych rozdzielczościach detale mogą zanikać, co może ułatwić lub utrudnić segmentację.
#   - Liczbę i kształt konturów: Mniejszy obraz może prowadzić do mniej precyzyjnych, bardziej "kanciastych" konturów lub łączenia bliskich obiektów.
#                          Większy obraz może dać więcej detali, ale też więcej szumu i fałszywych konturów.
#   - Parametry filtracji: Wartości MIN/MAX_CONTOUR_AREA muszą być dostosowane do nowej rozdzielczości, gdyż powierzchnie obiektów będą inne.
#   Czy zmniejszenie obrazu może poprawić detekcję (Zadanie 3b)?
#   - Tak, czasami. Może zredukować szum i drobne tekstury, upraszczając obraz i ułatwiając detekcję głównych obiektów.
#   - Jednak zbyt duże zmniejszenie prowadzi do utraty informacji i może pogorszyć detekcję małych obiektów lub rozróżnianie blisko położonych.
#   Eksperyment polega na modyfikacji TARGET_WIDTH i ponownym uruchomieniu skryptu.
print(f"\nZad.3: Wnioski dotyczące wpływu rozdzielczości znajdują się w komentarzach kodu.")


final_visualization_image = scaled_image_color.copy()
contour_filtering_visualization = scaled_image_color.copy()

stone_widths_px = []
stone_heights_px = []
stone_areas_px = []
processed_stone_counter = 0

print(f"\nUżywane filtry (Zad.6a): MIN_AREA={MIN_CONTOUR_AREA}, MAX_AREA={MAX_CONTOUR_AREA}")

# Zadanie 4. Numeryzacja kostek
# Zadanie 5. Pomiar wymiarów kostek
# Zadanie 6. Filtrowanie konturów po wielkości
for i, contour_candidate in enumerate(final_contours):
    # Zadanie 6a. Zaimplementuj filtrację konturów
    area = cv2.contourArea(contour_candidate)

    if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
        cv2.drawContours(contour_filtering_visualization, [contour_candidate], -1, (0, 255, 255), 1)
        continue

    processed_stone_counter += 1
    cv2.drawContours(contour_filtering_visualization, [contour_candidate], -1, (0, 255, 0), 2)

    # Zadanie 5a.i. Oblicz jej szerokość i wysokość w pikselach.
    x, y, w, h = cv2.boundingRect(contour_candidate)
    stone_widths_px.append(w)
    stone_heights_px.append(h)
    stone_areas_px.append(area)

    # Zadanie 4a. Zmodyfikuj pętlę iterującą po konturach tak, by na każdej kostce (nad jej środkiem) narysować numer porządkowy
    M = cv2.moments(contour_candidate)
    center_x, center_y = 0,0
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x = x + w // 2
        center_y = y + h // 2
    cv2.putText(final_visualization_image, f"K{processed_stone_counter}", (center_x - 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

    # Zadanie 5a.ii. Na oryginalnym obrazie narysuj prostokąt oraz opisz go wymiarami
    cv2.rectangle(final_visualization_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    dim_text = f"{w}x{h}px" # Zgodnie z przykładem "40x40 px"
    cv2.putText(final_visualization_image, dim_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Zadanie 4b. Dodaj zapisywanie każdej wyciętej kostki do osobnego pliku
    roi_bounding_box = scaled_image_color[y:y+h, x:x+w]
    if roi_bounding_box.size > 0:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"kostka_{processed_stone_counter:02d}.png"), roi_bounding_box)
    else:
        print(f"Ostrzeżenie: Pusty ROI dla kostki {processed_stone_counter}")

# Wnioski/Obserwacje (Zadanie 6b):
#   Zastosowanie filtracji konturów po wielkości (Zadanie 6a) pozwala na:
#   - Eliminację szumu: małe, nieistotne kontury powstałe w wyniku niedoskonałości progowania lub tekstury są usuwane.
#   - Eliminację niepożądanych obiektów: bardzo duże kontury, które mogą np. obejmować wiele kostek lub krawędź obrazu, są odrzucane.
#   Dobór progów MIN_CONTOUR_AREA i MAX_CONTOUR_AREA jest kluczowy i zależy od rozmiaru obiektów na obrazie
#   (po przeskalowaniu) oraz jakości progowania.
if processed_stone_counter == 0 and final_contours:
     print(f"INFO: Znaleziono {len(final_contours)} konturów, ale żaden nie przeszedł filtracji powierzchniowej (MIN:{MIN_CONTOUR_AREA}, MAX:{MAX_CONTOUR_AREA}).")

display_image("Wizualizacja filtrowania konturow (Zad.6)", contour_filtering_visualization, wait=False)
display_image("Finalna wizualizacja (Zad.4, Zad.5)", final_visualization_image, wait=False)

if processed_stone_counter > 0:
    print(f"Przetworzono i zapisano {processed_stone_counter} kostek do '{OUTPUT_DIR}'.")
else:
    print(f"Nie przetworzono żadnych kostek spełniających kryteria.")

# Zadanie 7. Liczenie i raportowanie kostek
# a. Na końcu całego procesu wyświetl w terminalu:
#    i. liczbę wykrytych kostek
#    ii. ich średnią szerokość i wysokość
#    iii. minimalny i maksymalny rozmiar
print(f"\n--- Zadanie 7. Raport ---")
if processed_stone_counter > 0:
    avg_width = np.mean(stone_widths_px)
    avg_height = np.mean(stone_heights_px)
    min_actual_area_found = np.min(stone_areas_px)
    max_actual_area_found = np.max(stone_areas_px)

    print(f"i.   Liczba wykrytych kostek: {processed_stone_counter}")
    print(f"ii.  Średnia szerokość: {avg_width:.2f} px, Średnia wysokość: {avg_height:.2f} px")
    print(f"iii. Minimalny rozmiar (powierzchnia): {min_actual_area_found:.2f} px^2")
    print(f"     Maksymalny rozmiar (powierzchnia): {max_actual_area_found:.2f} px^2")
else:
    print("Brak wykrytych kostek do zaraportowania.")

print("\nNaciśnij dowolny klawisz, aby zakończyć program.")
cv2.waitKey(0)
cv2.destroyAllWindows()