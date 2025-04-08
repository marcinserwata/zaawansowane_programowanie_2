import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_grouped_images(group_title, images, image_titles, cols=3, cmap=None):
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(group_title, fontsize=16)

    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()

    for i in range(n_images):
        if (cmap is None) and (images[i].ndim == 2):
            cmap_use = "gray"
        else:
            cmap_use = cmap
        axs[i].imshow(images[i], cmap=cmap_use)
        axs[i].set_title(image_titles[i])
        axs[i].axis('off')

    for j in range(n_images, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    # 1. Wstęp do progowania

    image = cv2.imread("image.jpg")
    if image is None:
        print("Błąd: Nie znaleziono obrazu 'image.jpg'. Upewnij się, że ścieżka jest poprawna.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T_values = [30, 100, 200]
    thresh_images = []
    thresh_titles = []
    for T in T_values:
        ret, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
        thresh_images.append(thresh)
        thresh_titles.append(f"T = {T}")

    images_task1 = [gray] + thresh_images
    titles_task1 = ["Skala szarości"] + thresh_titles
    show_grouped_images("1. Wstęp do progowania", images_task1, titles_task1, cols=2)

    # Komentarz:
    # Widać, że przy T=30 większość obrazu staje się biała (lub czarna), co może nie oddzielać wyraźnie obiektu od tła.
    # Natomiast przy T=200 tylko najjaśniejsze elementy pozostają w białej barwie.
    # W zależności od obrazu najlepsze efekty uzyskamy przy odpowiednio dobranym progu – często T=100 daje zbalansowany rezultat.

    # 2. Wpływ rozmycia

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh_blurred = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    ret, thresh_original = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    images_task2 = [blurred, thresh_blurred, thresh_original]
    titles_task2 = ["Obraz rozmyty", "Progowanie rozmyte (T=100)", "Progowanie oryginalne (T=100)"]
    show_grouped_images("2. Wpływ rozmycia", images_task2, titles_task2, cols=3)

    # Komentarz:
    # Rozmycie Gaussa wygładza obraz, co redukuje szumy. Dzięki temu progowanie
    # na rozmytym obrazie (T=100) może dać bardziej spójny efekt binarnego obrazu
    # niż progowanie oryginału, w którym mogą wystąpić małe zakłócające piksele.

    # 3. Efekty erozji

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh_blurred, kernel, iterations=1)
    images_task3 = [thresh_blurred, eroded]
    titles_task3 = ["Progowanie po rozmyciu (T=100)", "Po erozji"]
    show_grouped_images("3. Efekty erozji", images_task3, titles_task3, cols=2)

    # Komentarz:
    # Erozja powoduje "skurczenie" obszarów białych, co pozwala na usunięcie drobnych
    # szumów oraz niechcianych pikseli. Po operacji obrazy są czystsze, ale przy zbyt intensywnej erozji
    # możemy utracić ważne detale.

    # 4. Wpływ oświetlenia

    bright = cv2.add(gray, 50)
    ret, thresh_original_t4 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    ret, thresh_bright = cv2.threshold(bright, 100, 255, cv2.THRESH_BINARY)

    images_task4 = [gray, bright, thresh_original_t4, thresh_bright]
    titles_task4 = ["Oryginalny", "Rozjaśniony", "Progowanie oryginału (T=100)", "Progowanie rozjaśnionego (T=100)"]
    show_grouped_images("4. Wpływ oświetlenia", images_task4, titles_task4, cols=2)

    # Komentarz:
    # Zwiększenie jasności przesuwa histogram intensywności w stronę wyższych wartości.
    # W efekcie przy tym samym progu T=100 obraz rozjaśniony daje inny (często gorszy) efekt binarny,
    # co pokazuje, że metoda progowania jest bardzo czuła na zmiany oświetlenia.

    # 5. Progowanie metodą Otsu

    ret_otsu, otsu_thresh = cv2.threshold(bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images_task5 = [otsu_thresh]
    titles_task5 = [f"Otsu (T = {ret_otsu:.0f})"]
    show_grouped_images("5. Progowanie metodą Otsu", images_task5, titles_task5, cols=1)
    print("Wyznaczony próg przez Otsu:", ret_otsu)

    # Komentarz:
    # Metoda Otsu automatycznie wyznacza optymalny próg, analizując histogram obrazu.
    # Dzięki temu wynik progowania jest często lepszy od progowania z ustalonym T, szczególnie
    # przy zmiennych warunkach oświetleniowych.

    # 6. Analiza histogramu

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure(figsize=(6, 4))
    plt.plot(hist, color='gray')
    plt.title("6. Histogram skali szarości")
    plt.xlabel("Intensywność")
    plt.ylabel("Liczba pikseli")
    plt.axvline(x=ret_otsu, color='red', linestyle='dashed', linewidth=2)
    plt.text(ret_otsu + 5, max(hist) * 0.8, f"T = {ret_otsu:.0f}", color='red')
    plt.tight_layout()

    # Komentarz:
    # Histogram obrazu pokazuje rozkład intensywności pikseli.
    # Jeżeli na histogramie wyodrębnić są dwa główne skupiska – jedno odpowiada tłu, a drugie obiektowi,
    # to metoda Otsu znajduje próg minimalizujący wariancję między tymi dwoma grupami.

    # 7. Maska i segmentacja

    segmented = cv2.bitwise_and(image, image, mask=otsu_thresh)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    images_task7 = [otsu_thresh, segmented_rgb]
    titles_task7 = ["Maska Otsu", "Segmentacja (wycięty obiekt)"]
    show_grouped_images("7. Maska i segmentacja", images_task7, titles_task7, cols=2, cmap="gray")

    # Komentarz:
    # Segmentacja przy użyciu maski Otsu pozwala wyodrębnić obiekt, jednak jej skuteczność
    # zależy od jakości progowania. W przypadku złożonego tła lub słabych kontrastów mogą wystąpić
    # nieprecyzyjne krawędzie segmentowanego obiektu.

    # 8. Case study – kostka brukowa

    paving = cv2.imread("kostka_brukowa.jpg")
    if paving is None:
        print("Błąd: Nie znaleziono obrazu 'kostka_brukowa.jpg'. Upewnij się, że ścieżka jest poprawna.")
    else:
        paving_gray = cv2.cvtColor(paving, cv2.COLOR_BGR2GRAY)
        ret_paving_basic, paving_thresh_basic = cv2.threshold(paving_gray, 100, 255, cv2.THRESH_BINARY)
        ret_paving_otsu, paving_thresh_otsu = cv2.threshold(paving_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        paving_blurred = cv2.GaussianBlur(paving_gray, (5, 5), 0)
        ret_paving_otsu_blurred, paving_thresh_otsu_blurred = cv2.threshold(
            paving_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        paving_eroded = cv2.erode(paving_thresh_otsu_blurred, kernel, iterations=1)

        images_task8 = [
            paving_gray,
            paving_thresh_basic,
            paving_thresh_otsu,
            paving_thresh_otsu_blurred,
            paving_eroded
        ]
        titles_task8 = [
            "Skala szarości",
            "Progowanie podstawowe (T=100)",
            f"Progowanie Otsu (T={ret_paving_otsu:.0f})",
            "Otsu po rozmyciu",
            "Otsu+rozmycie+erozja"
        ]
        show_grouped_images("8. Case study – kostka brukowa", images_task8, titles_task8, cols=3)
        print("Wyznaczony próg przez Otsu dla kostki brukowej:", ret_paving_otsu)

        # Komentarz:
        # Porównanie progowania podstawowego z metodą Otsu wskazuje, że Otsu lepiej
        # dostosowuje się do lokalnych zmian intensywności na obrazie kostki brukowej.
        # Dodatkowo, rozmycie oraz erozja pomagają w eliminacji drobnych niedoskonałości,
        # co pozwala lepiej wychwycić wady powierzchni kostki brukowej.

    plt.show()


if __name__ == '__main__':
    main()
