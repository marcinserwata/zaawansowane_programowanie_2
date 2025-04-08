import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_text_image():
    img = np.zeros((300, 500), dtype=np.uint8)
    cv2.putText(img, 'SAMPLE TEXT', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 5, cv2.LINE_AA)
    return img


def create_captcha_image():
    img = np.zeros((200, 500), dtype=np.uint8)
    cv2.putText(img, 'CAPTCHA', (30, 150), cv2.FONT_HERSHEY_TRIPLEX, 3, 255, 5, cv2.LINE_AA)
    return img


def create_broken_text_image():
    img = np.zeros((150, 300), dtype=np.uint8)
    cv2.putText(img, 'CLOSING', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3, cv2.LINE_AA)
    cv2.line(img, (50, 30), (50, 120), 0, 5)
    cv2.line(img, (80, 30), (80, 120), 0, 5)
    cv2.line(img, (120, 30), (120, 120), 0, 5)
    cv2.line(img, (150, 30), (150, 120), 0, 5)
    return img


def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy


# 1. Analiza wpływu erozji na obrazy (tekst)

def erosion_analysis():
    img = create_text_image()

    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    eroded_rect = cv2.erode(img, kernel_rect, iterations=1)
    eroded_ellipse = cv2.erode(img, kernel_ellipse, iterations=1)

    titles = ['Oryginał (sample text)', 'Erozja (kernel kwadratowy)', 'Erozja (kernel eliptyczny)']
    images = [img, eroded_rect, eroded_ellipse]

    plt.figure(figsize=(12, 4))
    for i, disp_img in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(disp_img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(
        "Komentarz: Po erozji zauważamy wygładzenie krawędzi napisu, a drobne szczegóły (np. cienkie fragmenty liter) mogą być utracone. "
        "Kształt użytego kernela wpływa na sposób 'ścierania' liter.")


# 2. Eksperymentowanie z dylatacją

def dilation_experiment():
    img = np.zeros((200, 400), dtype=np.uint8)
    for i in range(10, 190, 20):
        cv2.line(img, (20, i), (380, i), 255, 1)

    iter_list = [1, 2, 3, 4, 5]
    dilated_images = []

    for it in iter_list:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(img, kernel, iterations=it)
        dilated_images.append(dilated)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, len(iter_list) + 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Oryginał')
    plt.axis('off')

    for i, dil_img in enumerate(dilated_images):
        plt.subplot(1, len(iter_list) + 1, i + 2)
        plt.imshow(dil_img, cmap='gray')
        plt.title(f'Iteracje: {iter_list[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(
        "Komentarz: Dylatacja powoduje powiększanie obiektów. Przy większej liczbie iteracji cienkie linie stają się grubsze, a przerwy pomiędzy nimi stopniowo zanikają.")


# 3. Usuwanie szumu za pomocą otwarcia

def noise_removal_opening():
    img = create_captcha_image()
    noisy = add_salt_pepper_noise(img, amount=0.1)

    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    ]
    opened_images = [cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k) for k in kernels]

    plt.figure(figsize=(14, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(noisy, cmap='gray')
    plt.title('Obraz zaszumiony')
    plt.axis('off')

    for i, op_img in enumerate(opened_images):
        plt.subplot(1, 4, i + 2)
        plt.imshow(op_img, cmap='gray')
        plt.title(f'Kernel: {kernels[i].shape[0]}x{kernels[i].shape[1]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("Komentarz: Operacja otwarcia redukuje szum solny i pieprzowy, usuwając drobne, niepożądane elementy, "
          "przy jednoczesnym zachowaniu ogólnego kształtu liter w obrazie CAPTCHA.")


# 4. Łączenie przerw w obiektach za pomocą zamknięcia

def closing_connection():
    img = create_broken_text_image()

    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    closing_rect = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
    closing_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Oryginał')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Z przerwami')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(closing_rect, cmap='gray')
    plt.title('Zamknięcie\n(prostokąt)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(closing_ellipse, cmap='gray')
    plt.title('Zamknięcie\n(elipsa)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(
        "Komentarz: Zwiększenie rozmiaru kernelu oraz liczby iteracji w operacji zamknięcia pozwoliło na lepsze połączenie fragmentów liter, poprawiając czytelność napisu 'CLOSING'.")


# 5. Eksperymentowanie z różnymi elementami strukturalnymi

def experiment_with_structural_elements():
    img = create_text_image()

    kernels = {
        'Prostokąt': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        'Krzyż': cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
        'Elipsa': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    }

    operations = {
        'Erozja': cv2.erode,
        'Dylatacja': cv2.dilate,
        'Otwarcie': lambda image, kernel: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        'Zamknięcie': lambda image, kernel: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        'Gradient': lambda image, kernel: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    }

    num_ops = len(operations)
    num_kernels = len(kernels)
    plt.figure(figsize=(15, 3 * num_ops))
    op_idx = 0

    for op_name, op_func in operations.items():
        col_idx = 0
        plt.subplot(num_ops, num_kernels + 1, op_idx * (num_kernels + 1) + col_idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Oryginał\n{op_name}')
        plt.axis('off')
        col_idx += 1
        for k_name, kernel in kernels.items():
            result = op_func(img, kernel)
            plt.subplot(num_ops, num_kernels + 1, op_idx * (num_kernels + 1) + col_idx + 1)
            plt.imshow(result, cmap='gray')
            plt.title(k_name)
            plt.axis('off')
            col_idx += 1
        op_idx += 1

    plt.tight_layout()
    plt.show()

    print("Komentarz: Efekty operacji morfologicznych różnią się w zależności od kształtu elementu strukturalnego. "
          "Kernel krzyżowy pozwala zachować inne szczegóły niż elementy prostokątne lub eliptyczne.")


# 6. Przykład zastosowania operacji morfologicznych (realny przypadek)

def real_world_application():
    img = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(img, 'ABC123', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3, cv2.LINE_AA)
    noisy_img = add_salt_pepper_noise(img, amount=0.08)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Tablica rejestracyjna')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Obraz zaszumiony')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(closed, cmap='gray')
    plt.title('Po operacjach')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Komentarz: W symulowanym przykładzie operacje morfologiczne usunęły szum oraz połączyły fragmenty liter, "
          "poprawiając czytelność tablicy rejestracyjnej.")


def main():
    print("1. Analiza wpływu erozji na obrazy (tekst)")
    erosion_analysis()

    print("\n2. Eksperymentowanie z dylatacją")
    dilation_experiment()

    print("\n3. Usuwanie szumu za pomocą otwarcia")
    noise_removal_opening()

    print("\n4. Łączenie przerw w obiektach za pomocą zamknięcia")
    closing_connection()

    print("\n5. Eksperymentowanie z różnymi elementami strukturalnymi")
    experiment_with_structural_elements()

    print("\n6. Przykład zastosowania operacji morfologicznych - poprawa czytelności tablic rejestracyjnych")
    real_world_application()


if __name__ == "__main__":
    main()
