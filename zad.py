import cv2
import numpy as np
import matplotlib.pyplot as plt


def task1_visualize_rgb_hsv():
    print("Task 1: Wizualizacja składowych RGB i HSV")
    image_path = "image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(rgb_image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_to_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    H, S, V = cv2.split(hsv_image)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title("RGB - oryginał")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(R, cmap='gray')
    axs[0, 1].set_title("Kanał R")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(G, cmap='gray')
    axs[0, 2].set_title("Kanał G")
    axs[0, 2].axis('off')

    axs[0, 3].imshow(B, cmap='gray')
    axs[0, 3].set_title("Kanał B")
    axs[0, 3].axis('off')

    axs[1, 0].imshow(hsv_to_rgb)
    axs[1, 0].set_title("HSV (do RGB)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(H, cmap='gray')
    axs[1, 1].set_title("Kanał H")
    axs[1, 1].axis('off')

    axs[1, 2].imshow(S, cmap='gray')
    axs[1, 2].set_title("Kanał S")
    axs[1, 2].axis('off')

    axs[1, 3].imshow(V, cmap='gray')
    axs[1, 3].set_title("Kanał V")
    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def task2_modify_saturation():
    print("Task 2: Modyfikacja saturacji")
    image_path = "image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)

    S_modified = cv2.add(S, 30)
    hsv_modified = cv2.merge([H, S_modified, V])
    modified_image = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(modified_rgb)
    axs[1].set_title("Saturacja +30")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def task3_detect_blue_objects():
    print("Task 3: Wykrywanie niebieskich obiektów")
    image_path = "blue_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(image, image, mask=mask_blue)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(res_blue, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(result_rgb)
    axs[1].set_title("Segmentacja niebieskiego")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def task4_change_hue():
    print("Task 4: Zmiana odcienia H")
    image_path = "image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)

    H_modified = ((H.astype(np.int32) + 30) % 180).astype(np.uint8)
    hsv_modified = cv2.merge([H_modified, S, V])
    modified_image = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(modified_rgb)
    axs[1].set_title("Zmiana odcienia (+30)")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def task5_detect_green_objects():
    print("Task 5: Wykrywanie zielonych obiektów")
    image_path = "green_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    res_green = cv2.bitwise_and(image, image, mask=mask_green)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(res_green, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(result_rgb)
    axs[1].set_title("Segmentacja zielonych")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def task6_detect_skin():
    print("Task 6: Rozpoznawanie koloru skóry")
    image_path = "portrait.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80])
    upper_skin = np.array([20, 255, 255])
    mask_skin = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask_skin)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(skin_rgb)
    axs[1].set_title("Wykrycie skóry")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def task7_modify_saturation():
    print("Task 7: Analiza nasycenia")
    image_path = "image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_image)

    S_decreased = cv2.subtract(S, 50)
    hsv_decreased = cv2.merge([H, S_decreased, V])
    image_decreased = cv2.cvtColor(hsv_decreased, cv2.COLOR_HSV2BGR)

    S_increased = cv2.add(S, 50)
    hsv_increased = cv2.merge([H, S_increased, V])
    image_increased = cv2.cvtColor(hsv_increased, cv2.COLOR_HSV2BGR)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    decreased_rgb = cv2.cvtColor(image_decreased, cv2.COLOR_BGR2RGB)
    increased_rgb = cv2.cvtColor(image_increased, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(decreased_rgb)
    axs[1].set_title("Nasycenie -50")
    axs[1].axis('off')

    axs[2].imshow(increased_rgb)
    axs[2].set_title("Nasycenie +50")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


def task8_multi_color_segmentation():
    print("Task 8: Segmentacja kolorów z wieloma maskami")
    image_path = "multicolor_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Błąd podczas wczytywania obrazu:", image_path)
        return

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    combined_mask = cv2.bitwise_or(mask_blue, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red)

    result = cv2.bitwise_and(image, image, mask=combined_mask)

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Oryginalny obraz")
    axs[0].axis('off')

    axs[1].imshow(result_rgb)
    axs[1].set_title("Segmentacja kolorów")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task1_visualize_rgb_hsv()
    task2_modify_saturation()
    task3_detect_blue_objects()
    task4_change_hue()
    task5_detect_green_objects()
    task6_detect_skin()
    task7_modify_saturation()
    task8_multi_color_segmentation()
