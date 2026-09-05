import cv2
import numpy as np
from ultralytics import YOLO

# 1. Загружаем модель сегментации (лучше "yolov8n-seg.pt" для скорости)
model = YOLO('yolov8n-seg.pt') 

# Список классов, которые являются ПРЕПЯТСТВИЯМИ (COCO датасет)
OBSTACLE_CLASSES = [0, 1, 2, 3, 5, 7] # 0=человек, 1=велосипед, 2=машина, 3=мотоцикл, 5=автобус, 7=грузовик
# Если робот едет в помещении - добавьте 56 (стул), 60 (обеденный стол) и т.д.

def is_path_clear(frame, roi_percent=0.4, threshold=0.08):
    """
    roi_percent: какая часть кадра считается "зоной прямо перед роботом" (нижние 40%)
    threshold: при какой доле перекрытия (0.08 = 8%) останавливаться
    """
    height, width, _ = frame.shape
    
    # 2. Задаем зону интереса (ROI) - только нижняя часть кадра (куда едет робот)
    roi_y_start = int(height * (1 - roi_percent))  # Отступ снизу
    roi = frame[roi_y_start:height, 0:width]
    
    # 3. Запускаем инференс (сегментацию)
    results = model(roi, conf=0.4, iou=0.5, verbose=False)  # conf - порог уверенности
    
    # 4. Создаем пустую черную маску для препятствий
    obstacle_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    
    # 5. Обрабатываем найденные маски
    if results[0].masks is not None:
        for mask, cls in zip(results[0].masks.data, results[0].boxes.cls):
            class_id = int(cls.item())
            if class_id in OBSTACLE_CLASSES:
                # Конвертируем тензор маски в бинарную маску OpenCV
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (roi.shape[1], roi.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                # Добавляем к общей маске препятствий
                obstacle_mask = cv2.bitwise_or(obstacle_mask, binary_mask)
    
    # 6. Вычисляем процент площади, занятой препятствиями в ROI
    total_pixels = roi.shape[0] * roi.shape[1]
    obstacle_pixels = np.sum(obstacle_mask)
    obstacle_ratio = obstacle_pixels / total_pixels
    
    # 7. Бинарное решение
    is_clear = obstacle_ratio < threshold
    
    # ---- Визуализация для отладки (можно убрать) ----
    # Накладываем маску на кадр
    roi_colored = roi.copy()
    roi_colored[obstacle_mask == 1] = (0, 0, 255)  # Красный цвет - препятствие
    frame[roi_y_start:height, 0:width] = roi_colored
    
    # Пишем статус
    status_text = "GO (True)" if is_clear else "STOP (False)"
    color = (0, 255, 0) if is_clear else (0, 0, 255)
    cv2.putText(frame, f"{status_text} (Blocked: {obstacle_ratio*100:.1f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # -------------------------------------------------
    
    return is_clear, frame, obstacle_ratio

# ----- Основной цикл с камеры -----
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Переворачиваем зеркально для удобства (опционально)
    frame = cv2.flip(frame, 1)
    
    # Получаем решение
    can_go, display_frame, ratio = is_path_clear(frame)
    
    # Вывод в консоль
    print(f"Можно ехать: {can_go} | Занято: {ratio*100:.2f}%")
    
    # Показываем окно
    cv2.imshow("Robot Vision", display_frame)
    
    # Выход по клавише Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
