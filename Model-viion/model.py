# detect_realtime.py
from ultralytics import YOLO
import cv2
import time
from datetime import datetime

# Загружаем модель
model = YOLO("yolov8n.pt")  # или yolov8s.pt для лучшей точности

TARGET_CLASSES = {
    "person": "человек",
    "chair": "стул",
    "dining table": "стол",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "cell phone": "телефон",
    "book": "книга",
    "bottle": "бутылка",
    "cup": "чашка",
    "couch": "диван",
}

def detect_realtime(source=0):
    """
    source: 
        0 - веб-камера по умолчанию
        1, 2, ... - другие камеры
        "rtsp://..." - RTSP поток
        "http://..." - HTTP поток
        "video.mp4" - видео файл
    """
    
    # Открываем источник видео
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Ошибка: не удалось открыть камеру/поток")
        return
    
    # Настраиваем параметры камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    
    print(" Детекция запущена. Нажмите 'q' для выхода, 's' для скриншота")
    print("─" * 50)
    
    # Статистика
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Не удалось получить кадр")
                break
            
            frame_count += 1
            
            # Детекция объектов (каждый кадр)
            # Для ускорения можно обрабатывать каждый 2-3 кадр
            results = model(frame, verbose=False)
            
            # Рисуем результаты
            annotated = results[0].plot()
            
            # Считаем FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Добавляем информацию на кадр
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Получаем информацию об объектах
            boxes = results[0].boxes
            detected_objects = {}
            
            for box in boxes:
                cls_id = int(box.cls[0])
                label_en = model.names[cls_id]
                label_ru = TARGET_CLASSES.get(label_en, label_en)
                
                if label_ru not in detected_objects:
                    detected_objects[label_ru] = 0
                detected_objects[label_ru] += 1
            
            # Выводим статистику в консоль (каждые 60 кадров)
            if frame_count % 60 == 0 and detected_objects:
                print(f"\n [{datetime.now().strftime('%H:%M:%S')}]")
                for obj, count in detected_objects.items():
                    print(f"   {obj}: {count}")
            
            # Показываем кадр
            cv2.imshow("Real-time Detection", annotated)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Выход...")
                break
            elif key == ord('s'):
                # Скриншот
                screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, annotated)
                print(f"📸 Скриншот сохранён: {screenshot_path}")
            elif key == ord('p'):
                # Пауза
                print("⏸️  Пауза. Нажмите любую клавишу для продолжения...")
                cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("\n👋 Прервано пользователем")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Обработано кадров: {frame_count}")


if __name__ == "__main__":
    import sys
    
    # Выбор источника
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # Пробуем преобразовать в int (номер камеры)
        try:
            source = int(source)
        except ValueError:
            pass  # оставляем как строку (URL)
    else:
        print("Выберите источник:")
        print("  0 - Веб-камера по умолчанию")
        print("  1, 2, ... - Другие камеры")
        print("  RTSP/HTTP URL - Сетевой поток")
        print("  video.mp4 - Видео файл")
        print("\nНажмите Enter для веб-камеры или введите путь/номер:")
        
        source_input = input("> ").strip()
        if not source_input:
            source = 0
        else:
            try:
                source = int(source_input)
            except ValueError:
                source = source_input
    
    detect_realtime(source)