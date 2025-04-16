import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

kostka_brukowa = 'kostka_brukowa.jpg'
dokument = 'dokument.jpg'
image = 'image.jpg'

def display_images(images, titles, rows, cols, main_title="Porównanie wyników"):
    plt.figure(figsize=(cols * 4, rows * 4))
    plt.suptitle(main_title, fontsize=16)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("--- Część 1: Porównanie metod progowania ---")
if os.path.exists(kostka_brukowa):
    img_uneven_gray = cv2.imread(kostka_brukowa, cv2.IMREAD_GRAYSCALE)

    if img_uneven_gray is None:
        print(f"Błąd: Nie można wczytać obrazu '{kostka_brukowa}'. Sprawdź ścieżkę i format pliku.")
    else:
        T = 100
        ret_simple, thresh_simple = cv2.threshold(img_uneven_gray, T, 255, cv2.THRESH_BINARY)
        ret_otsu, thresh_otsu = cv2.threshold(img_uneven_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        block_size_adaptive = 11
        C_adaptive = 2
        thresh_adaptive_mean = cv2.adaptiveThreshold(img_uneven_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY, block_size_adaptive, C_adaptive)
        thresh_adaptive_gauss = cv2.adaptiveThreshold(img_uneven_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, block_size_adaptive, C_adaptive)

        images_to_display = [
            img_uneven_gray, thresh_simple, thresh_otsu,
            thresh_adaptive_mean, thresh_adaptive_gauss
        ]
        titles = [
            'Oryginał (szary)', f'Proste (T={T})', f'Otsu (T={int(ret_otsu)})',
            'Adaptacyjne (Mean)', 'Adaptacyjne (Gaussian)'
        ]
        display_images(images_to_display, titles, 2, 3, "1. Porównanie Metod Progowania")

        print("\n1. Wnioski:")
        print("- Proste progowanie (globalne) działa słabo przy nierównym oświetleniu. Jasne obszary stają się całkowicie białe, ciemne - czarne, tracąc szczegóły.")
        print("- Progowanie Otsu znajduje optymalny próg globalny, ale nadal ma problemy z lokalnymi zmianami oświetlenia. Jest lepsze niż ręczne ustawienie progu, ale niewystarczające dla tego obrazu.")
        print("- Progowanie adaptacyjne (Mean i Gaussian) radzi sobie znacznie lepiej.")
        print("- Metoda Gaussian często daje nieco gładsze wyniki i jest mniej wrażliwa na szum niż Mean.")
        print("=> Najlepiej z nierównym światłem poradziły sobie metody ADAPTACYJNE.")
else:
    print(f"Błąd: Plik obrazu '{kostka_brukowa}' nie został znaleziony. Nie można uruchomić części 1.")


print("\n--- Część 2: Wpływ rozmiaru sąsiedztwa (blockSize) ---")
if os.path.exists(kostka_brukowa):
    if 'img_uneven_gray' not in locals() or img_uneven_gray is None:
         img_uneven_gray = cv2.imread(kostka_brukowa, cv2.IMREAD_GRAYSCALE)
         if img_uneven_gray is None:
              print(f"Błąd: Nie można wczytać obrazu '{kostka_brukowa}' do części 2.")

    if 'img_uneven_gray' in locals() and img_uneven_gray is not None:
        block_sizes = [11, 21, 31, 41]
        adaptive_results_bs = []
        titles_bs = ['Oryginał (szary)']
        images_bs = [img_uneven_gray]
        C_bs = 2

        for bs in block_sizes:
            thresh = cv2.adaptiveThreshold(img_uneven_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, bs, C_bs)
            adaptive_results_bs.append(thresh)
            titles_bs.append(f'Gaussian, blockSize={bs}')
            images_bs.append(thresh)

        display_images(images_bs, titles_bs, 2, 3, f"2. Wpływ blockSize (Gaussian, C={C_bs})")

        print("\n2. Wnioski:")
        print("- Mały `blockSize` (np. 11): Dobrze adaptuje się do bardzo lokalnych zmian, ale może wzmacniać szum i drobne tekstury. Krawędzie mogą być 'poszarpane'.")
        print("- Średni `blockSize` (np. 21, 31): Często stanowi dobry kompromis. Wygładza drobny szum, zachowując główne kontury.")
        print("- Duży `blockSize` (np. 41): Daje gładsze wyniki, ale może zatracić drobniejsze szczegóły lub źle działać, gdy znacząca zmiana oświetlenia występuje w obrębie jednego bloku. Przy bardzo dużych wartościach zaczyna przypominać progowanie globalne.")
        print("=> Optymalny `blockSize` zależy od skali detali, które chcemy wykryć, oraz od skali zmian oświetlenia. Dla kostki brukowej, wartości 21 lub 31 mogą być dobrym punktem wyjścia.")
elif not os.path.exists(kostka_brukowa):
     print(f"Błąd: Plik obrazu '{kostka_brukowa}' nie został znaleziony. Nie można uruchomić części 2.")


print("\n--- Część 3: Różne metody adaptacyjne i parametr C ---")
if os.path.exists(kostka_brukowa):
    if 'img_uneven_gray' not in locals() or img_uneven_gray is None:
         img_uneven_gray = cv2.imread(kostka_brukowa, cv2.IMREAD_GRAYSCALE)
         if img_uneven_gray is None:
              print(f"Błąd: Nie można wczytać obrazu '{kostka_brukowa}' do części 3.")

    if 'img_uneven_gray' in locals() and img_uneven_gray is not None:
        C_values = [2, 5, 10, 15]
        block_size_C = 21
        results_mean_C = []
        results_gauss_C = []
        titles_C = ['Oryginał (szary)']
        images_C_mean = [img_uneven_gray]
        images_C_gauss = [img_uneven_gray]

        for C in C_values:
            thresh_mean = cv2.adaptiveThreshold(img_uneven_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, block_size_C, C)
            results_mean_C.append(thresh_mean)
            images_C_mean.append(thresh_mean)
            titles_C.append(f'Mean, C={C}')

            thresh_gauss = cv2.adaptiveThreshold(img_uneven_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, block_size_C, C)
            results_gauss_C.append(thresh_gauss)
            images_C_gauss.append(thresh_gauss)

        display_images(images_C_mean, titles_C, 2, 3, f"3. Wpływ C dla metody Mean (blockSize={block_size_C})")
        display_images(images_C_gauss, titles_C, 2, 3, f"3. Wpływ C dla metody Gaussian (blockSize={block_size_C})")

        print("\n3. Wnioski:")
        print("- Małe `C` (np. 2, 5): Powoduje, że próg jest bliższy lokalnej średniej/ważonej średniej. Więcej pikseli może zostać sklasyfikowanych jako białe (pierwszy plan). Może to uwydatnić słabe krawędzie, ale też szum.")
        print("- Duże `C` (np. 10, 15): Zwiększa próg (bo odejmujemy większą wartość). Mniej pikseli staje się białych. Może pomóc w eliminacji szumu i drobnych elementów tła, ale może też usunąć słabsze, ale istotne elementy.")
        print("- Metoda `Gaussian` jest generalnie mniej wrażliwa na lokalne artefakty i szum niż `Mean`, ponieważ używa ważonej średniej (piksele bliżej centrum mają większy wpływ).")
        print("=> Lepsze radzenie sobie z szumem i nierównym tłem:")
        print("  - Metoda: `ADAPTIVE_THRESH_GAUSSIAN_C` jest często preferowana.")
        print("  - Parametr C: Dobór zależy od konkretnego obrazu i celu. Wyższe `C` lepiej tłumią szum tła kosztem potencjalnej utraty detali. Trzeba eksperymentować, ale wartości C w zakresie 2-10 są często dobrym punktem startowym.")
elif not os.path.exists(kostka_brukowa):
     print(f"Błąd: Plik obrazu '{kostka_brukowa}' nie został znaleziony. Nie można uruchomić części 3.")


print("\n--- Część 4: Segmentacja tekstu w dokumencie ---")
if os.path.exists(dokument):
    img_doc_color = cv2.imread(dokument)
    if img_doc_color is None:
        print(f"Błąd: Nie można wczytać obrazu '{dokument}'.")
    else:
        img_doc_gray = cv2.cvtColor(img_doc_color, cv2.COLOR_BGR2GRAY)
        doc_block_size = 15
        doc_C = 7
        thresh_doc = cv2.adaptiveThreshold(img_doc_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, doc_block_size, doc_C)
        thresh_doc_inv = cv2.adaptiveThreshold(img_doc_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, doc_block_size, doc_C)

        display_images(
            [img_doc_gray, thresh_doc, thresh_doc_inv],
            ['Oryginał (szary)', 'Tekst (Biały na czarnym)', 'Tekst (Czarny na białym - INV)'],
            1, 3, "4. Segmentacja Tekstu w Dokumencie"
        )
        print("\n4. Wnioski:")
        print("- Wyświetlono obraz binarny. Progowanie adaptacyjne dobrze radzi sobie z nierównym oświetleniem typowym dla zdjęć dokumentów.")
else:
     print(f"Błąd: Plik obrazu '{dokument}' nie został znaleziony. Nie można uruchomić części 4.")


print("\n--- Część 5: Automatyczna maska ROI ---")
if os.path.exists(image):
    img_objects_color = cv2.imread(image)
    if img_objects_color is None:
         print(f"Błąd: Nie można wczytać obrazu '{image}'.")
    else:
        img_objects_gray = cv2.cvtColor(img_objects_color, cv2.COLOR_BGR2GRAY)
        obj_block_size = 31
        obj_C = 5
        mask_obj = cv2.adaptiveThreshold(img_objects_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, obj_block_size, obj_C)

        kernel = np.ones((3,3), np.uint8)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel, iterations=2)

        result_roi = cv2.bitwise_and(img_objects_color, img_objects_color, mask=mask_obj)

        display_images(
            [img_objects_color, mask_obj, result_roi],
            ['Oryginał (kolor)', 'Wygenerowana Maska', 'Obiekty (ROI)'],
            1, 3, "5. Automatyczna Maska ROI"
        )
        print("\n5. Wnioski:")
        print("- Progowanie adaptacyjne (ew. w połączeniu z morfologią) pozwala na wygenerowanie maski obiektów.")
        print("- Operacja `bitwise_and` z użyciem maski skutecznie izoluje obiekty (ROI) z oryginalnego obrazu.")

else:
     print(f"Błąd: Plik obrazu '{image}' nie został znaleziony. Nie można uruchomić części 5.")


print("\n--- Część 6: Interaktywna analiza parametrów ---")

if os.path.exists(image):
    img_interactive_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    if img_interactive_gray is None:
        print(f"Błąd: Nie można wczytać obrazu '{image}' do interaktywnej analizy.")
    else:
        window_name = 'Interaktywne Progowanie Adaptacyjne (Gaussian)'
        cv2.namedWindow(window_name)

        initial_block_size = 11
        initial_C = 2

        def update_threshold(*args):
            block_size = cv2.getTrackbarPos('blockSize', window_name)
            C = cv2.getTrackbarPos('C', window_name) - 20

            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1

            thresh_interactive = cv2.adaptiveThreshold(img_interactive_gray, 255,
                                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                       cv2.THRESH_BINARY, block_size, C)

            cv2.imshow(window_name, thresh_interactive)

        cv2.createTrackbar('blockSize', window_name, (initial_block_size - 3) // 2 , 25, update_threshold)
        cv2.createTrackbar('C', window_name, initial_C + 20, 40, update_threshold)

        update_threshold(0)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            try:
                 if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                 break

        cv2.destroyAllWindows()
        print("\n. Wnioski:")
        print("- Interaktywne narzędzie pozwala efektywnie dobrać parametry `blockSize` i `C`.")
        print("- Wizualna ocena wyniku w czasie rzeczywistym jest kluczowa dla znalezienia optymalnych ustawień dla danego obrazu.")


else:
    print(f"Błąd: Plik obrazu '{image}' nie został znaleziony. Nie można uruchomić części 6.")

print("\n--- Zakończono przetwarzanie ---")