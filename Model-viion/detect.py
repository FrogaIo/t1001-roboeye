# detect_zones.py
from ultralytics import YOLO
import cv2
import time
import numpy as np

class ZoneDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        
        self.ALLOWED_CLASSES = {
            "person": "person",
            "chair": "chair",
            "dining table": "dining table",
            "cell phone": "cell phone",
        }
        
        self.control_zone = [0, 240, 640, 480]
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        self.temp_zone = None
        
        self.KNOWN_HEIGHTS = {
            "person": 170,
            "chair": 90,
            "dining table": 75,
            "cell phone": 15,
        }
        
    def set_zone(self, x1, y1, x2, y2):
        self.control_zone = [x1, y1, x2, y2]
        print(f"Zone set: ({x1},{y1}) -> ({x2},{y2})")
        
    def is_in_zone(self, bbox):
        if not self.control_zone:
            return False
        
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = self.control_zone
        
        cx = (x1 + x2) / 2
        bottom_y = y2
        
        in_x = zx1 <= cx <= zx2
        in_y = zy1 <= bottom_y <= zy2
        
        return in_x and in_y
    
    def calculate_distance(self, bbox, label):
        x1, y1, x2, y2 = bbox
        obj_height_px = y2 - y1
        
        if label in self.KNOWN_HEIGHTS and obj_height_px > 0:
            real_height_cm = self.KNOWN_HEIGHTS[label]
            focal_length = 125
            distance = (real_height_cm * focal_length) / obj_height_px
            return distance
        else:
            frame_height = 480
            bottom_y = y2
            distance = (1 - bottom_y / frame_height) * 100
            return distance
    
    def get_zone_distance(self, bbox):
        if not self.control_zone:
            return 0
        
        x1, y1, x2, y2 = bbox
        zx1, zy1, zx2, zy2 = self.control_zone
        
        return abs(y2 - zy1)
    
    def process(self, frame):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_time)
            self.fps_time = time.time()
        
        results = self.model(frame, verbose=False)
        annotated = frame.copy()
        
        # Рисуем зону
        if self.control_zone:
            zx1, zy1, zx2, zy2 = self.control_zone
            overlay = annotated.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 100, 255), -1)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(annotated, "ZONE", (zx1 + 5, zy1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.temp_zone:
            tx1, ty1, tx2, ty2 = self.temp_zone
            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
        
        objects_in_zone_close = 0  # в зоне и < 1м
        objects_in_zone_far = 0    # в зоне и > 1м
        objects = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]
            
            if label not in self.ALLOWED_CLASSES:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            in_zone = self.is_in_zone(bbox)
            distance = self.calculate_distance(bbox, label)
            zone_dist = self.get_zone_distance(bbox)
            
            # Цвет рамки
            if in_zone:
                if distance <= 100:  # В зоне и меньше 1 метра
                    color = (0, 0, 255)  # Красный
                    dist_label = "CLOSE"
                    objects_in_zone_close += 1
                else:  # В зоне и больше 1 метра
                    color = (0, 255, 255)  # Желтый
                    dist_label = "FAR"
                    objects_in_zone_far += 1
            else:
                color = (0, 255, 0)  # Зеленый
                dist_label = "OUTSIDE"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Подпись с расстоянием
            label_text = f"{label} {conf:.2f} {distance:.0f}cm"
            if in_zone:
                label_text += f" [{dist_label}]"
            
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            if in_zone:
                zone_info = f"zone_dist: {zone_dist}px"
                cv2.putText(annotated, zone_info, (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            objects.append({
                'class': label,
                'conf': conf,
                'in_zone': in_zone,
                'distance': distance,
                'zone_dist': zone_dist,
                'dist_label': dist_label
            })
        
        # Статус: STOP только если есть объект в зоне ближе 1 метра
        if objects_in_zone_close > 0:
            status = "STOP"
            status_color = (0, 0, 255)
        else:
            status = "GO"
            status_color = (0, 255, 0)
        
        cv2.rectangle(annotated, (420, 10), (630, 70), (0, 0, 0), -1)
        cv2.rectangle(annotated, (420, 10), (630, 70), status_color, 3)
        cv2.putText(annotated, status, (440, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        cv2.putText(annotated, f"FPS: {self.fps:.0f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Отладка
        if self.frame_count % 30 == 0:
            print(f"\n--- Frame {self.frame_count} ---")
            for obj in objects:
                zone_str = ">>> IN ZONE <<<" if obj['in_zone'] else "outside"
                print(f"  {obj['class']}: {obj['conf']:.2f} - {obj['distance']:.0f}cm - {obj['dist_label']} - {zone_str}")
            print(f"Close (<1m): {objects_in_zone_close} | Far (>1m): {objects_in_zone_far} -> Status: {status}")
        
        return annotated


def main():
    detector = ZoneDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    
    print("=== Zone Control System ===")
    print("Detecting: person, chair, dining table, cell phone")
    print("Mouse drag = draw zone | c = clear | r = reset | q = quit | s = screenshot")
    
    drawing = False
    start_pt = None
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start_pt
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing and start_pt:
            detector.temp_zone = (start_pt[0], start_pt[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP and drawing and start_pt:
            drawing = False
            detector.temp_zone = None
            x1 = min(start_pt[0], x)
            y1 = min(start_pt[1], y)
            x2 = max(start_pt[0], x)
            y2 = max(start_pt[1], y)
            
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                detector.set_zone(x1, y1, x2, y2)
            else:
                print("Zone too small, ignored")
    
    cv2.namedWindow("Zone Control")
    cv2.setMouseCallback("Zone Control", mouse_cb)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated = detector.process(frame)
            cv2.imshow("Zone Control", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.control_zone = None
                print("Zone cleared")
            elif key == ord('r'):
                detector.control_zone = [0, 240, 640, 480]
                print("Zone reset")
            elif key == ord('s'):
                path = f"shot_{int(time.time())}.jpg"
                cv2.imwrite(path, annotated)
                print(f"Saved: {path}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()