import cv2
import imutils
import numpy as np

# Zadanie 1 - Wykrywanie logo w butelce Fanty
# a. Wczytaj obraz butelki i szablon logo.
fanta_image = cv2.imread('fanta_butelka.jpg')
fanta_logo = cv2.imread('fanta_logo.jpg')
# b. Wykonaj cv2.matchTemplate.
fanta_gray = cv2.cvtColor(fanta_image, cv2.COLOR_BGR2GRAY)
logo_gray = cv2.cvtColor(fanta_logo, cv2.COLOR_BGR2GRAY)
result1 = cv2.matchTemplate(fanta_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
# c. Zaznacz wykryte logo ramką.
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result1)
startX, startY = max_loc
endX = startX + fanta_logo.shape[1]
endY = startY + fanta_logo.shape[0]
cv2.rectangle(fanta_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
# d. Wskaż współrzędne wykrytego logo i wartość dopasowania (maxVal).
print(f"Zadanie 1 - Współrzędne: ({startX}, {startY}), maxVal: {max_val:.4f}")
# Wniosek: Logo wykryto z wysoką zgodnością (maxVal: 0.9993) w koordynatach (174, 141).
cv2.imshow("Zadanie 1 - Wykryte logo Fanty", fanta_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zadanie 2 - Wrażliwość na obrót
for angle in [30, 45]:
    # a. Obróć obraz Fanty o 30° i 45°.
    rotated = imutils.rotate(fanta_image, angle)
    # b. Powtórz detekcję logo.
    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    result2 = cv2.matchTemplate(rotated_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)
    startX2, startY2 = max_loc2
    endX2 = startX2 + fanta_logo.shape[1]
    endY2 = startY2 + fanta_logo.shape[0]
    cv2.rectangle(rotated, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
    # c. Zinterpretuj wynik: czy detekcja zadziałała? Jaki jest maxVal?
    print(f"Zadanie 2 (obrót {angle}°) - Współrzędne: ({startX2}, {startY2}), maxVal: {max_val2:.4f}")
    if angle == 30:
        # Wniosek: Dla obrotu 30° wartość maxVal 0.2587 – niskie dopasowanie, detekcja nie jest skuteczna przy tym kącie.
        pass
    else:
        # Wniosek: Dla obrotu 45° wartość maxVal 0.3082 – nieznaczna poprawa, ale nadal słabe dopasowanie.
        pass
    cv2.imshow(f"Zadanie 2 - Obrót {angle}°", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Zadanie 3 - Wrażliwość na zmianę rozmiaru szablonu
# a. Zmniejsz lub powiększ oryginalny obraz Fanty.
scaled_up = cv2.resize(fanta_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
scaled_down = cv2.resize(fanta_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
for label, img in [("Powiększony 1.5x", scaled_up), ("Zmniejszony 0.5x", scaled_down)]:
    # b. Wykonaj dopasowanie tym samym szablonem.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result3 = cv2.matchTemplate(gray_img, logo_gray, cv2.TM_CCOEFF_NORMED)
    min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(result3)
    startX3, startY3 = max_loc3
    endX3 = startX3 + fanta_logo.shape[1]
    endY3 = startY3 + fanta_logo.shape[0]
    cv2.rectangle(img, (startX3, startY3), (endX3, endY3), (0, 0, 255), 2)
    # c. Zaobserwuj, czy detekcja zadziałała poprawnie.
    print(f"Zadanie 3 ({label}) - Współrzędne: ({startX3}, {startY3}), maxVal: {max_val3:.4f}")
    if label == "Powiększony 1.5x":
        # Wniosek: Dla obrazu powiększonego 1.5x maxVal 0.4144 – umiarkowane dopasowanie, wykrycie mniej precyzyjne.
        pass
    else:
        # Wniosek: Dla obrazu zmniejszonego 0.5x maxVal 0.2990 – niskie dopasowanie, detekcja trudna przy mniejszej skali.
        pass
    cv2.imshow(f"Zadanie 3 - {label}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Zadanie 4 - Porównanie metod cv2.matchTemplate
methods = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
}
base_image = cv2.imread('fanta_butelka.jpg')
base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
best_normed = ("", -1.0)
best_sqdiff = ("", 1.0)
for name, method in methods.items():
    # a. Użyj metody.
    result4 = cv2.matchTemplate(base_gray, logo_gray, method)
    # b. Narysuj wykryty prostokąt.
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(result4)
        top_left = min_loc4
        match_val = min_val4
    else:
        min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(result4)
        top_left = max_loc4
        match_val = max_val4
    bottom_right = (top_left[0] + fanta_logo.shape[1], top_left[1] + fanta_logo.shape[0])
    vis = base_image.copy()
    cv2.rectangle(vis, top_left, bottom_right, (0, 255, 255), 2)
    # c. Wypisz wartości dopasowania.
    print(f"Zadanie 4 ({name}) - matchVal: {match_val:.4f}")
    if name.endswith("NORMED") and "SQDIFF" not in name:
        if match_val > best_normed[1]:
            best_normed = (name, match_val)
    if name == "TM_SQDIFF_NORMED":
        if match_val < best_sqdiff[1]:
            best_sqdiff = (name, match_val)
    cv2.imshow(f"Zadanie 4 - {name}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Wniosek: Najlepszą metodą dla dopasowania normalizowanego jest TM_CCORR_NORMED (matchVal: 0.9999), a w przypadku SQDIFF_NORMED najniższy matchVal: 0.0003.

# Zadanie 5 - Detekcja małych ikon interfejsu
# a. Wczytaj zrzut ekranu i ikonę jako szablon.
screenshot = cv2.imread('lupa.png')
icon_template = cv2.imread('lupa_guzik.png')
screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
icon_gray = cv2.cvtColor(icon_template, cv2.COLOR_BGR2GRAY)
# b. Wykonaj dopasowanie na pełnym zrzucie.
result5 = cv2.matchTemplate(screenshot_gray, icon_gray, cv2.TM_CCOEFF_NORMED)
min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(result5)
# c. Zaznacz wynik i porównaj z oczekiwanym.
startX5, startY5 = max_loc5
endX5 = startX5 + icon_template.shape[1]
endY5 = startY5 + icon_template.shape[0]
cv2.rectangle(screenshot, (startX5, startY5), (endX5, endY5), (255, 0, 255), 2)
print(f"Zadanie 5 - Współrzędne: ({startX5}, {startY5}), maxVal: {max_val5:.4f}")
# Wniosek: Ikona wykryta dokładnie z maksymalną wartością 1.0000, co oznacza perfekcyjne dopasowanie.
cv2.imshow("Zadanie 5 - Detekcja ikony", screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zadanie 6 - Odporność na "fałszywe trafienia"
# a. Wczytaj obraz z wieloma podobnymi obiektami i wytnij jeden jako szablon.
lego_image = cv2.imread('figury.jpg')
lego_template = cv2.imread('trojkat.jpg')
lego_gray = cv2.cvtColor(lego_image, cv2.COLOR_BGR2GRAY)
template_gray6 = cv2.cvtColor(lego_template, cv2.COLOR_BGR2GRAY)
# b. Spróbuj wykryć jego wystąpienia.
result6 = cv2.matchTemplate(lego_gray, template_gray6, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
locs = np.where(result6 >= threshold)
vis6 = lego_image.copy()
detections = 0
for pt in zip(*locs[::-1]):
    cv2.rectangle(vis6, pt, (pt[0] + lego_template.shape[1], pt[1] + lego_template.shape[0]), (0, 165, 255), 2)
    detections += 1
# c. Wypisz liczbę wykryć i czy były błędne.
print(f"Zadanie 6 - Liczba wykryć powyżej progu {threshold}: {detections}")
# Wniosek: Znaleziono 2980 wykryć – duża liczba fałszywych trafień wymaga podniesienia progu lub dodatkowych kryteriów.
vis6_small = cv2.resize(vis6, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("Zadanie 6 - Wyniki dopasowania LEGO", vis6_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zadanie 7 - Połączenie konturów i template matchingu
# a. Wczytaj obraz z wieloma obiektami i znajdź kontury.
multi_obj = cv2.imread('figury.jpg')
gray_multi = cv2.cvtColor(multi_obj, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_multi, (5, 5), 0)
_, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# b. Dla każdego konturu wytnij fragment i wykonaj template matching.
pattern = cv2.imread('trojkat.jpg')
pattern_gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
matched_vis = multi_obj.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = multi_obj[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Skalowanie ROI, jeśli jest mniejsze niż wzorzec
    if roi_gray.shape[0] < pattern_gray.shape[0] or roi_gray.shape[1] < pattern_gray.shape[1]:
        continue
    res7 = cv2.matchTemplate(roi_gray, pattern_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val7, _, max_loc7 = cv2.minMaxLoc(res7)
    # c. Zidentyfikuj podobne obiekty (próg 0.7).
    if max_val7 >= 0.7:
        cv2.rectangle(matched_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f"Zadanie 7 - Kontur w ({x}, {y}, {w}, {h}) – matchVal: {max_val7:.4f}")
matched_vis_small = cv2.resize(matched_vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("Zadanie 7 - Wyniki konturów i template matchingu", matched_vis_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
