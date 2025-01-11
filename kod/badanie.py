
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pyzbar.pyzbar import decode
import time
import csv

def filter_finder_patterns(finder_patterns_bounds, image_size, threshold_ratio=0.1):
    """
    Filtruje obiekty, które nie są częścią kodu QR, na podstawie wielkości i położenia.

    finder_patterns_bounds -- Lista prostokątnych obwiedni (x, y, w, h).
    image_size -- Rozmiar obrazu (szerokość, wysokość).
    threshold_ratio -- Maksymalny stosunek wielkości obiektu do obrazu (domyślnie 10%).
    """
    max_area = (image_size[0] * image_size[1]) * threshold_ratio
    filtered_patterns = []
    
    for x, y, w, h in finder_patterns_bounds:
        area = w * h
        if area < max_area:  
            filtered_patterns.append((x, y, w, h))
    
    return filtered_patterns

def draw_finder_patterns(image, bounds):
    """
    Rysuje prostokąty Finder Patterns na obrazie.
    :param image: Obraz wejściowy.
    :param bounds: Lista prostokątów (x, y, w, h).
    :return: Obraz z naniesionymi prostokątami.
    """
    output_image = image.copy()
    for i, (x, y, w, h) in enumerate(bounds):
        
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        text = f"{w}x{h}"
        cv2.putText(output_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output_image

def calculate_qr_and_finder_pattern_sizes(image, bounds):
    """
    Oblicza rozmiar całego kodu QR oraz zewnętrznych Finder Patterns.
    :param image: Obraz przyciętego kodu QR.
    :param bounds: Lista prostokątów (x, y, w, h) dla Finder Patterns.
    :return: Długość kodu QR (width, height) oraz maksymalne wymiary Finder Patterns.
    """
    
    qr_height, qr_width = image.shape[:2]

   
    max_finder_width = max(w for _, _, w, _ in bounds)
    max_finder_height = max(h for _, _, _, h in bounds)

    return (qr_width, qr_height), (max_finder_width, max_finder_height)

def get_qr_code_corners(finder_patterns_bounds):
    all_corners = []
    for x, y, w, h in finder_patterns_bounds:
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        all_corners.extend(corners)
    min_x = min(corner[0] for corner in all_corners)
    max_x = max(corner[0] for corner in all_corners)
    min_y = min(corner[1] for corner in all_corners)
    max_y = max(corner[1] for corner in all_corners)
    return (min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)

def determine_grid_size(finder_width, qr_width, possible_sizes):
    """
    Określa rozmiar siatki (Grid Size) na podstawie proporcji Finder Pattern.
    :param finder_width: Szerokość Finder Pattern w pikselach.
    :param qr_width: Szerokość kodu QR w pikselach.
    :param possible_sizes: Lista możliwych rozmiarów siatki (np. [21, 25, 29, ...]).
    :return: Rozmiar siatki (Grid Size).
    """
    
    proportion = finder_width / qr_width
    predicted_grid_size = round(7 / proportion)
    closest_size = min(possible_sizes, key=lambda x: abs(x - predicted_grid_size))
    return closest_size


def highlight_finder_patterns(image, modules):
    """
    Zaznacza Finder Patterns na obrazie kodu QR.

    Args:
    image -- Obraz kodu QR w skali szarości lub kolorowy (numpy array).
    modules -- Tablica 2D reprezentująca siatkę modułów kodu QR.

    Returns:
    overlay -- Obraz z zaznaczonymi Finder Patterns.
    """

    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
   
    finder_coordinates = [
        (slice(0, 7), slice(0, 7)),      
        (slice(0, 7), slice(-7, None)),  
        (slice(-7, None), slice(0, 7))   
    ]

   
    module_size = image.shape[0] // modules.shape[0]

  
    for coords in finder_coordinates:
        y_range, x_range = coords

      
        y_start = y_range.start if y_range.start is not None else modules.shape[0]
        y_stop = y_range.stop if y_range.stop is not None else modules.shape[0]
        x_start = x_range.start if x_range.start is not None else modules.shape[1]
        x_stop = x_range.stop if x_range.stop is not None else modules.shape[1]

     
        overlay[y_start * module_size:y_stop * module_size, x_start * module_size:x_stop * module_size] = (0, 255, 0)  # Zielony

    return overlay

def segment_qr_code(image, grid_size):
    height, width = image.shape[:2]
    module_size = height // grid_size
    modules = []
    for row in range(grid_size):
        row_modules = []
        for col in range(grid_size):
            module = image[row * module_size:(row + 1) * module_size, col * module_size:(col + 1) * module_size]
            mean_intensity = np.mean(module)
            row_modules.append(1 if mean_intensity < 128 else 0)
        modules.append(row_modules)
    return np.array(modules)

def crop_qr_code(image, corners):
    x_min = min(corners[0][0], corners[2][0])
    x_max = max(corners[1][0], corners[3][0])
    y_min = min(corners[0][1], corners[1][1])
    y_max = max(corners[2][1], corners[3][1])
    return image[y_min:y_max, x_min:x_max]




def highlight_separators(image, modules):
    """
    Zaznacza na obrazie Separatory wokół Finder Patterns w kodzie QR.

    Args:
    image -- Obraz kodu QR w skali szarości lub kolorowy (numpy array).
    modules -- Tablica 2D reprezentująca siatkę modułów kodu QR.

    Returns:
    overlay -- Obraz z zaznaczonymi Separatorami.
    """
   
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size

    
    separator_coords = [
       
        (slice(0, 7), slice(7, 8)),  
        (slice(7, 8), slice(0, 8)),  
        
        (slice(0, 7), slice(grid_size - 8, grid_size - 7)),  
        (slice(7, 8), slice(grid_size - 8, grid_size)),  
        
        (slice(grid_size - 8, grid_size - 7), slice(0, 8)),  
        (slice(grid_size - 7, grid_size), slice(7, 8))  
    ]

    
    for coord in separator_coords:
        y_range, x_range = coord
        overlay[
            y_range.start * module_size : y_range.stop * module_size,
            x_range.start * module_size : x_range.stop * module_size,
        ] = (0, 255, 255)  

    return overlay


def highlight_timing_patterns(image, modules):
    """
    Zaznacza na obrazie Timing Patterns w kodzie QR.

    Args:
    image -- Obraz kodu QR w skali szarości lub kolorowy (numpy array).
    modules -- Tablica 2D reprezentująca siatkę modułów kodu QR.

    Returns:
    overlay -- Obraz z zaznaczonymi Timing Patterns.
    """
    
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size

 
    for col in range(8, grid_size - 8):
        x_start = col * module_size
        y_start = 6 * module_size
        x_end = x_start + module_size
        y_end = y_start + module_size
        overlay[y_start:y_end, x_start:x_end] = (255, 0, 0)  

  
    for row in range(8, grid_size - 8):
        x_start = 6 * module_size
        y_start = row * module_size
        x_end = x_start + module_size
        y_end = y_start + module_size
        overlay[y_start:y_end, x_start:x_end] = (255, 0, 0)  

    return overlay




def highlight_format_information(image, modules):
    """
    Podświetla sekcje Format Information na obrazie QR.

    Args:
    image -- Przycięty obraz kodu QR (numpy array).
    modules -- Tablica modułów QR (1 = czarny, 0 = biały).

    Returns:
    overlay -- Obraz z podświetlonymi sekcjami Format Information.
    """
    
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
   
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size

   
    format_info_coords = [
        (slice(0, 9), slice(8, 9)),  
        (slice(8, 9), slice(0, 9)),  
        (slice(-8, -1), slice(8, 9)),  
        (slice(8, 9), slice(-8, -1)),  
    ]
    
   
    for coord in format_info_coords:
        y_range, x_range = coord
        overlay[y_range.start * module_size:y_range.stop * module_size,
                x_range.start * module_size:x_range.stop * module_size] = (128, 0, 128) 

    return overlay


def highlight_data_modules_corrected(image, modules):
    """
    Zaznacza na obrazie QR kodu obszar danych (Data Modules) zgodnie z wymaganiami standardu QR Code w wersji 1.

    Arguments:
    image -- Przycięty obraz QR Code (numpy array).
    modules -- 2D array reprezentujący siatkę modułów QR Code.

    Returns:
    overlay -- Obraz z zaznaczonym obszarem danych.
    """
   
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size

    
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    
    for row in range(grid_size):
        for col in range(grid_size):
           
            if (
                not (row < 9 and col < 9)  
                and not (row < 9 and col >= grid_size - 8) 
                and not (row >= grid_size - 8 and col < 9)  
                and not (row == 6 or col == 6)  
                and not (row == 8 and (col < 9 or col >= grid_size - 8))  
                and not (col == 8 and (row < 9 or row >= grid_size - 8))  
                and not (row == 8 and col == 13)  
            ):
                
                x1 = col * module_size
                y1 = row * module_size
                x2 = x1 + module_size
                y2 = y1 + module_size

                
                color = (0, 0, 255)  
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  

    return overlay


def highlight_all_modules(image, modules):
    """
    Nakłada wszystkie warstwy (Finder Patterns, Separatory, Timing Patterns, Format Information, Data Modules) 
    na obraz QR Code.

    Arguments:
    image -- Przycięty obraz QR Code (numpy array).
    modules -- 2D array reprezentująca siatkę modułów QR Code.

    Returns:
    overlay -- Obraz z zaznaczonymi wszystkimi komponentami QR Code.
    """
   
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

    
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size

    
    finder_coordinates = [
        (slice(0, 7), slice(0, 7)),      
        (slice(0, 7), slice(-7, None)),  
        (slice(-7, None), slice(0, 7))   
    ]
    for coords in finder_coordinates:
        y_range, x_range = coords
        y_start = (y_range.start or 0) * module_size
        y_stop = (y_range.stop or grid_size) * module_size
        x_start = (x_range.start or 0) * module_size
        x_stop = (x_range.stop or grid_size) * module_size
        overlay[y_start:y_stop, x_start:x_stop] = (0, 255, 0) 

    
    separator_coords = [
        
        (slice(0, 7), slice(7, 8)),  
        (slice(7, 8), slice(0, 8)),  
        
        (slice(0, 7), slice(grid_size - 8, grid_size - 7)),
        (slice(7, 8), slice(grid_size - 8, grid_size)),
        
        (slice(grid_size - 8, grid_size - 7), slice(0, 8)),
        (slice(grid_size - 7, grid_size), slice(7, 8))
    ]
    for coord in separator_coords:
        y_range, x_range = coord
        y_start = y_range.start * module_size
        y_stop = y_range.stop * module_size
        x_start = x_range.start * module_size
        x_stop = x_range.stop * module_size
        overlay[y_start:y_stop, x_start:x_stop] = (0, 255, 255)  

    
    for i in range(8, grid_size - 8):
        overlay[6 * module_size:(6 + 1) * module_size, i * module_size:(i + 1) * module_size] = (255, 0, 0)  
        overlay[i * module_size:(i + 1) * module_size, 6 * module_size:(6 + 1) * module_size] = (255, 0, 0)  #

    
    format_info_coords = [
        (slice(0, 9), slice(8, 9)),  
        (slice(8, 9), slice(0, 9)),  
        (slice(-8, -1), slice(8, 9)),  
        (slice(8, 9), slice(-8, -1)) 
    ]
    for coord in format_info_coords:
        y_range, x_range = coord
        y_start = y_range.start * module_size
        y_stop = y_range.stop * module_size
        x_start = x_range.start * module_size
        x_stop = x_range.stop * module_size
        overlay[y_start:y_stop, x_start:x_stop] = (128, 0, 128)  

   
    for row in range(grid_size):
        for col in range(grid_size):
            if (
                not (row < 9 and col < 9) and  
                not (row < 9 and col >= grid_size - 8) and  
                not (row >= grid_size - 8 and col < 9) and  
                not (row == 6 or col == 6) and  
                not (row == 8 and (col < 9 or col >= grid_size - 8)) and  
                not (col == 8 and (row < 9 or row >= grid_size - 8)) and  
                not (row == grid_size - 8 and col == 8)  
            ):
                x1, y1 = col * module_size, row * module_size
                x2, y2 = x1 + module_size, y1 + module_size
                overlay[y1:y2, x1:x2] = (0, 0, 255)  

    return overlay




def generate_image_from_matrix(matrix):
    """
    Tworzy obraz QR z macierzy 0 i 1.
    
    Args:
    matrix -- Lista list (0 i 1) reprezentująca QR Code.

    Returns:
    img -- Obraz Pillow.
    """
    size = len(matrix)
    
    img = Image.fromarray(np.array(matrix) * 255).convert('L')  
    img = img.resize((size * 10, size * 10), Image.NEAREST)  
    return img


def read_qr_from_image(image):
    """
    Odczytuje dane z obrazu QR Code za pomocą pyzbar.
    
    Args:
    image -- Obraz Pillow (QR Code).

    Returns:
    Decoded message or None.
    """
    decoded_objects = decode(image)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')  
    return None

def invert_matrix(matrix):
    """
    Inverts a binary matrix (0 -> 1, 1 -> 0).

    Arguments:
    matrix -- List of lists containing binary values.

    Returns:
    inverted_matrix -- Inverted binary matrix.
    """
    inverted_matrix = [[1 - value for value in row] for row in matrix]
    return inverted_matrix


def visualize_modules_with_colored_values_no_text(image, modules):
    grid_size = modules.shape[0]
    module_size = image.shape[0] // grid_size
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * module_size
            y1 = row * module_size
            x2 = x1 + module_size
            y2 = y1 + module_size
            value = modules[row, col]
            color = (255, 0, 0) if value == 1 else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)






def process_all_qr_codes(folder_path):
    possible_sizes = [21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
                       65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105,
                         109, 113, 117, 121, 125, 129, 133, 137, 141,
                           145, 149, 153, 157, 161, 165, 169, 173, 177]
    total_files = 0
    correct_count = 0
    incorrect_count = 0
    decoded_results = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_files += 1
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            finder_patterns = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    if area > 100:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        if 0.9 < aspect_ratio < 1.1:
                            finder_patterns.append(approx)

            finder_patterns_bounds = [cv2.boundingRect(pattern) for pattern in finder_patterns]

            pattern_image = image.copy()
            for x, y, w, h in finder_patterns_bounds:
                cv2.rectangle(pattern_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            
            image_size = (binary_image.shape[1], binary_image.shape[0])
            filtered_finder_patterns_bounds = filter_finder_patterns(finder_patterns_bounds, image_size)



            pattern_image_filtered = image.copy()
            for x, y, w, h in filtered_finder_patterns_bounds:
                cv2.rectangle(pattern_image_filtered, (x, y), (x + w, y + h), (0, 0, 255), 2)

            corners_image = pattern_image_filtered.copy()

            for x, y, w, h in filtered_finder_patterns_bounds:
                corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]  
                for corner in corners:
                    cv2.circle(corners_image, corner, 5, (0, 255, 0), -1)  

            

            qr_size, max_finder_size = calculate_qr_and_finder_pattern_sizes(corners_image, filtered_finder_patterns_bounds)

            qr_corners = get_qr_code_corners(filtered_finder_patterns_bounds)

            qr_corner_image = pattern_image_filtered.copy()
            for corner in qr_corners:
                cv2.circle(qr_corner_image, corner, 10, (255, 0, 0), -1)  
            
            cropped_qr_image = crop_qr_code(pattern_image_filtered, qr_corners)

            qr_height, qr_width = cropped_qr_image.shape[:2]

            finders_proportion = max_finder_size[0] / qr_height

            grid_size = determine_grid_size(max_finder_size[0], qr_width, possible_sizes)

            if len(cropped_qr_image.shape) == 3:
                cropped_qr_gray = cv2.cvtColor(cropped_qr_image, cv2.COLOR_BGR2GRAY)
            else:
                cropped_qr_gray = cropped_qr_image

            modules = segment_qr_code(cropped_qr_gray, grid_size)

            qr_matrix = invert_matrix(modules)
            inverted_qr_matrix = invert_matrix(qr_matrix)

            qr_image = generate_image_from_matrix(qr_matrix)
            
            decoded_message = read_qr_from_image(qr_image)

             # Sprawdzenie poprawności wiadomości na podstawie nazwy pliku
            expected_message = filename.split('-')[0]  # Pobranie wiadomości z nazwy pliku
            if decoded_message == expected_message:
                validation_result = "Poprawna"
                correct_count += 1
            else:
                validation_result = "Niepoprawna"
                incorrect_count += 1

            decoded_results.append((filename, expected_message, decoded_message if decoded_message else "", validation_result))
        
    print(f"Łączna liczba plików: {total_files}")
    print(f"Poprawnie odczytane wiadomości: {correct_count}")
    print(f"Niepoprawnie odczytane wiadomości: {incorrect_count}")

    return decoded_results, total_files, correct_count, incorrect_count


def main():

    folder_path = "qr_dataset"
    start_time = time.time()  # Rozpoczęcie pomiaru czasu
    results, total_files, correct_count, incorrect_count = process_all_qr_codes(folder_path)
    end_time = time.time()  # Zakończenie pomiaru czasu

    # Zapis wyników do pliku CSV
    csv_file_path = os.path.join(folder_path, "qr_code_results.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Nazwa pliku", "Oczekiwana wiadomość", "Odczytana wiadomość", "Ocena"])
        for result in results:
            writer.writerow(result)

    print(f"Wyniki zapisano do pliku: {csv_file_path}")
    print(f"Czas przetwarzania: {end_time - start_time:.2f} sekundy")
    print(f"Liczba plików w folderze: {total_files}")
    print(f"Poprawnie odczytane wiadomości: {correct_count}")
    print(f"Niepoprawnie odczytane wiadomości: {incorrect_count}")

if __name__ =="__main__":
    main()












