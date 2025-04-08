#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_images_group(main_title, images, cols=3, figsize=(15, 10)):
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    plt.figure(figsize=figsize)
    plt.suptitle(main_title, fontsize=16)
    for i, (title, img) in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img is None:
            plt.text(0.5, 0.5, "Brak obrazu", horizontalalignment='center', verticalalignment='center')
        else:
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# Zadanie 1: Eksploracja różnych metod rozmycia
def task1_exploration(image):
    ksize = (5, 5)
    median_ksize = 5  # musi być nieparzyste
    blur_simple = cv2.blur(image, ksize)
    blur_gauss = cv2.GaussianBlur(image, ksize, 0)
    blur_median = cv2.medianBlur(image, median_ksize)
    blur_bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Grupujemy wyniki w jednej figurze
    images = [
        ("Oryginalny obraz", image),
        ("cv2.blur", blur_simple),
        ("GaussianBlur", blur_gauss),
        ("MedianBlur", blur_median),
        ("BilateralFilter", blur_bilateral)
    ]
    show_images_group("Zadanie 1: Eksploracja metod rozmycia", images, cols=3)

    # Komentarze:
    # i. Najlepsze usuwanie szumów: cv2.bilateralFilter skutecznie redukuje szum przy zachowaniu krawędzi.
    # ii. Najlepsze zachowanie szczegółów: bilateral filter, dzięki uwzględnieniu odległości i różnic pikseli.
    # iii. Zalety i wady:
    #      - cv2.blur: Prosty, szybki; ale traci szczegóły.
    #      - GaussianBlur: Naturalne rozmycie, dobre wygładzanie; przy dużych kernelach – nadmierne wygładzanie.
    #      - MedianBlur: Skuteczny przy szumie impulsowym; wymaga nieparzystego kernela.
    #      - BilateralFilter: Najlepsze zachowanie krawędzi, ale większe obciążenie obliczeniowe.


# Zadanie 2: Analiza wpływu rozmiaru kernela
def task2_kernel_effect(image):
    kernel_sizes = [3, 5, 9, 15]
    for k in kernel_sizes:
        kernel = (k, k)
        median_ksize = k if k % 2 == 1 else k + 1

        if k % 2 == 0:
            gauss_kernel = (k + 1, k + 1)
        else:
            gauss_kernel = (k, k)

        simple = cv2.blur(image, kernel)
        gauss = cv2.GaussianBlur(image, gauss_kernel, 0)
        median = cv2.medianBlur(image, median_ksize)
        d = k if k % 2 == 1 else k + 1
        bilateral = cv2.bilateralFilter(image, d=d, sigmaColor=75, sigmaSpace=75)

        images = [
            (f"Blur {kernel}", simple),
            (f"Gaussian {gauss_kernel}", gauss),
            (f"Median {median_ksize}", median),
            (f"Bilateral d={d}", bilateral)
        ]
        show_images_group(f"Zadanie 2: Kernel size: {kernel}", images, cols=2, figsize=(10, 6))

    # Komentarz:
    # i. Większy kernel → większe wygładzenie, ale potencjalnie utrata detali.
    # ii. Kernel (5x5) lub (9x9) to dobry kompromis.


# Zadanie 3: Rozmycie dwustronne w praktyce
def task3_bilateral(image):
    noisy = image.copy().astype(np.float32)
    noise = np.zeros_like(noisy)
    cv2.randn(noise, 0, 25)
    noisy = cv2.add(noisy, noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    params = [
        {"d": 5, "sigmaColor": 50, "sigmaSpace": 50},
        {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
        {"d": 15, "sigmaColor": 100, "sigmaSpace": 100}
    ]
    bilateral_images = []
    for p in params:
        img_bilat = cv2.bilateralFilter(noisy, d=p["d"],
                                        sigmaColor=p["sigmaColor"],
                                        sigmaSpace=p["sigmaSpace"])
        title = f"Bilateral d={p['d']}, σC={p['sigmaColor']}"
        bilateral_images.append((title, img_bilat))

    comp_images = [
        ("Zaszumiony obraz", noisy),
        ("GaussianBlur", cv2.GaussianBlur(noisy, (9, 9), 0))
    ]

    show_images_group("Zadanie 3: Bilateral Filter (różne parametry)", bilateral_images, cols=2, figsize=(12, 6))
    show_images_group("Zadanie 3: Porównanie GaussianBlur z zaszumionym obrazem", comp_images, cols=2, figsize=(10, 5))

    # Komentarz:
    # Rozmycie dwustronne świetnie redukuje szum i zachowuje ostre krawędzie.


# Zadanie 4: Rozmycie na obrazie zawierającym tekst
def task4_text_blur():
    text_image_path = "text_image.png"
    if not os.path.isfile(text_image_path):
        print(f"Nie znaleziono obrazu {text_image_path}.")
        return
    text_img = cv2.imread(text_image_path)

    results = [
        ("Oryginalny tekst", text_img),
        ("cv2.blur", cv2.blur(text_img, (5, 5))),
        ("GaussianBlur", cv2.GaussianBlur(text_img, (5, 5), 0)),
        ("MedianBlur", cv2.medianBlur(text_img, 5)),
        ("BilateralFilter", cv2.bilateralFilter(text_img, d=9, sigmaColor=75, sigmaSpace=75))
    ]
    show_images_group("Zadanie 4: Rozmycie na obrazie z tekstem", results, cols=3, figsize=(12, 6))

    # Komentarz:
    # W metodach blur i GaussianBlur tekst traci ostrość, natomiast w median i bilateral
    # filtru litery są bardziej czytelne.


# Zadanie 5: Porównanie skuteczności redukcji szumów
def task5_noise_reduction(image):
    noisy = image.copy().astype(np.float32)
    noise = np.zeros_like(noisy)
    cv2.randn(noise, 0, 30)
    noisy = cv2.add(noisy, noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    results = [
        ("Zaszumiony obraz", noisy),
        ("cv2.blur", cv2.blur(noisy, (5, 5))),
        ("GaussianBlur", cv2.GaussianBlur(noisy, (5, 5), 0)),
        ("MedianBlur", cv2.medianBlur(noisy, 5)),
        ("BilateralFilter", cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75))
    ]
    show_images_group("Zadanie 5: Redukcja szumów", results, cols=3, figsize=(12, 6))

    # Komentarz:
    # Bilateral i median są często lepsze przy usuwaniu szumów zachowując detale.


# Zadanie 6: Symulacja efektu głębi ostrości
def task6_depth_of_field(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = w // 4, h // 4
    x2, y2 = x1 + w // 2, y1 + h // 2
    mask[y1:y2, x1:x2] = 255

    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    depth_of_field = cv2.add(background, foreground)

    results = [
        ("Oryginalny obraz", image),
        ("Maska obiektu", mask),
        ("Symulacja DoF", depth_of_field)
    ]
    show_images_group("Zadanie 6: Symulacja efektu głębi ostrości", results, cols=3, figsize=(14, 5))

    # Komentarz:
    # Ręczne maskowanie pozwala na selektywne rozmycie tła, co daje efekt podobny do głębi ostrości w fotografii.


def main():
    sample_image_path = "input.png"
    if not os.path.isfile(sample_image_path):
        print(f"Nie znaleziono obrazu {sample_image_path}.")
        return
    image = cv2.imread(sample_image_path)

    print("=== Zadanie 1: Eksploracja metod rozmycia ===")
    task1_exploration(image)

    print("=== Zadanie 2: Analiza wpływu rozmiaru kernela ===")
    task2_kernel_effect(image)

    print("=== Zadanie 3: Rozmycie dwustronne w praktyce ===")
    task3_bilateral(image)

    print("=== Zadanie 4: Rozmycie na obrazie zawierającym tekst ===")
    task4_text_blur()

    print("=== Zadanie 5: Porównanie redukcji szumów ===")
    task5_noise_reduction(image)

    print("=== Zadanie 6: Symulacja efektu głębi ostrości ===")
    depth_image_path = "depth_image.jpg"
    if not os.path.isfile(depth_image_path):
        print(f"Nie znaleziono obrazu {depth_image_path}.")
    else:
        depth_img = cv2.imread(depth_image_path)
        task6_depth_of_field(depth_img)


if __name__ == "__main__":
    main()
