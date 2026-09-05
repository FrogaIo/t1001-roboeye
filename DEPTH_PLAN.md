# План интеграции monocular depth в RoboEye

Этот документ предназначен для следующего разработчика или AI-агента. Текущая
система уже передаёт JPEG с телефона на Mac по WebSocket, запускает YOLO-World,
рисует перспективную проезжую зону и возвращает `GO`/`STOP` с маршрутом.

## 1. Рекомендуемая модель

Использовать `depth-anything/Depth-Anything-V2-Small-hf`.

Почему:

- 24.8 млн параметров — самая лёгкая официальная версия Depth Anything V2;
- лицензия Apache-2.0;
- запускается через Hugging Face Transformers без копирования исходников модели;
- PyTorch умеет выполнять её на `mps` GPU Apple Silicon;
- выдаёт плотную карту глубины для всего изображения, включая неизвестные YOLO
  объекты.

Ограничение: это **относительная глубина**. Значение нельзя честно подписывать
как метры без калибровки или metric-варианта модели. Для RoboEye сначала нужно
решать более надёжную задачу: «есть ли поверхность заметно ближе ожидаемой
плоскости пола в данной полосе».

## 2. Отдельный smoke test модели

Рекомендуется Python 3.12 или 3.13. Не ломать рабочее окружение RoboEye до
успешного отдельного теста.

```bash
cd /Users/roman/Documents/GitHub/t1001-roboeye
python3.12 -m venv .venv-depth
source .venv-depth/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision transformers pillow safetensors \
  opencv-python numpy
```

Создать временный `depth_smoke.py` со следующим содержимым:

```python
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT = Path("sample.jpg")
OUTPUT = Path("depth-preview.jpg")

device = "mps" if torch.backends.mps.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

image = Image.open(INPUT).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

with torch.inference_mode():
    prediction = model(**inputs).predicted_depth
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]

depth = prediction.float().cpu().numpy()
low, high = np.percentile(depth, (2, 98))
normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
preview = cv2.applyColorMap(
    (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
)
cv2.imwrite(str(OUTPUT), preview)

print(f"device={device} shape={depth.shape} range=({depth.min():.3f}, {depth.max():.3f})")
print(f"saved {OUTPUT}")
```

Положить рядом обычный JPEG `sample.jpg`, затем запустить:

```bash
python depth_smoke.py
open depth-preview.jpg
```

Обязательно проверить экспериментом, в какую сторону модели соответствует
«ближе»: поставить одну коробку рядом, вторую далеко и сравнить медианы внутри
их областей. Не фиксировать знак глубины по предположению.

## 3. Изменения архитектуры

Добавить файл `depth_detector.py` с классом `DepthDetector`:

```text
DepthDetector.__init__
  ├── выбирает mps/cpu
  ├── один раз загружает processor
  └── один раз загружает model и вызывает eval()

DepthDetector.infer(frame_bgr)
  ├── BGR → RGB/PIL
  ├── inference_mode()
  ├── resize результата до размера кадра
  └── возвращает float32 H×W без цветовой визуализации
```

Модель нельзя загружать внутри обработки каждого кадра. YOLO и depth должны
исполняться последовательно в существующем worker thread: конкурентный запуск
двух моделей на одном MPS добавит нестабильность и почти наверняка не ускорит
демо.

Рекомендуемые изменения файлов:

- `requirements.txt`: добавить `transformers`, `pillow`, `safetensors`;
- `config.py`: ID модели, период depth-инференса и пороги остатка;
- `depth_detector.py`: загрузка модели и получение карты;
- `app.py`: запуск depth не чаще 2 раз/с и повторное использование последней
  карты между вычислениями;
- `detector.py`: преобразование depth в занятость перспективных полос;
- `static/monitor.html`: переключатель обычного изображения/depth overlay и
  численные lane scores;
- `static/app.js`: оставить только контуры опасных областей и маршрут, не
  отправлять плотную depth map в JSON.

## 4. Как извлечь препятствия без классов

Нельзя просто взять «самые близкие 20% пикселей»: нижняя часть самого пола всегда
будет близкой и превратится в ложное препятствие.

Минимально жизнеспособный алгоритм:

1. Ограничить анализ текущей перспективной трапецией RoboEye.
2. Уменьшить depth map, например до `160×90`.
3. Стабилизировать её экспоненциальным средним по времени.
4. Для каждой строки трапеции оценить фон пола медианой или нижним перцентилем
   значений по всей ширине проезжей зоны.
5. Вычесть этот профиль из depth map. Положительный или отрицательный знак
   выбрать по результату smoke test «ближняя/дальняя коробка».
6. Нормировать остаток через MAD/перцентили, а не абсолютное значение модели.
7. Применить morphological open/close и удалить маленькие компоненты.
8. Для каждой компоненты взять нижнюю центральную точку и определить полосу тем
   же `projected_lane()`, который уже использует RoboEye.
9. Считать полосу занятой, если площадь компоненты и её depth residual превышают
   настроенные пороги.

Псевдокод:

```python
roi_depth = depth[trapezoid_mask]
floor_profile[y] = percentile(depth[y, corridor_left:corrridor_right], 35)
residual[y, x] = signed_near(depth[y, x] - floor_profile[y])
obstacle_mask = residual > adaptive_threshold
obstacle_mask = morphology(obstacle_mask)
components = connected_components(obstacle_mask)
```

Если коробка перекрывает почти всю ширину коридора и ломает row median, брать
фон пола из истории чистых кадров или из боковых 15% трапеции.

## 5. Объединение с YOLO

Depth не заменяет текущий детектор, а закрывает его слепую зону.

```text
depth-компонента в CENTER ───────────────► STOP: UNKNOWN OBSTACLE
          │
          └── пересекается с YOLO box ──► STOP: BOX / PERSON / ...

нет depth-препятствия + нет YOLO danger ─► GO после 3 чистых кадров
```

Правила:

- безопасность определяет depth occupancy;
- YOLO даёт понятную человеку причину;
- близкий `person` из YOLO может по-прежнему вызывать немедленный `STOP`, даже
  если depth в этом кадре неуверен;
- маршрут выбирается по объединению занятых полос depth и YOLO;
- при устаревшей depth map старше 1 секунды не использовать её как свежую;
- при исключении модели не падать: залогировать ошибку и продолжить в текущем
  YOLO-only режиме.

## 6. Частота и производительность

Начальные настройки:

```text
YOLO:             до 3 Гц, как сейчас
Depth Anything:   2 Гц
Depth input:      стандартный размер processor, затем попробовать 392 px
Temporal EMA:     alpha 0.35
Max depth age:    1.0 сек
```

Сначала замерить 30 прогретых запусков и записать median/p95. Если combined p95
выше 400 мс:

1. снизить частоту depth до 1.5 Гц;
2. уменьшить вход depth;
3. выполнять YOLO и depth поочерёдно, сохраняя последние результаты;
4. только после этого рассматривать официальный Core ML Small-вариант Apple.

Не начинать с Core ML: Transformers проще интегрировать и отлаживать. Core ML —
отдельный этап оптимизации после доказанного алгоритма проезжаемости.

## 7. Интерфейс

Телефон:

- живое RGB-видео остаётся основным;
- поверх — контуры depth-препятствий, рамки YOLO и одна траектория;
- не показывать полноэкранную радужную heatmap: она закрывает сцену;
- добавить короткий источник решения: `DEPTH`, `YOLO` или `DEPTH + YOLO`.

Mac:

- переключатель `RGB / DEPTH / FUSED`;
- возраст последней depth map;
- время depth inference;
- scores трёх полос;
- reason и снимок инцидента сохранять как сейчас;
- для отладки можно сохранять рядом с incident JPEG маленькую depth preview.

## 8. Проверки и критерии приёмки

Unit tests:

- одинаковая синтетическая плоскость не создаёт препятствие;
- близкий прямоугольник в каждой полосе занимает только эту полосу;
- объект за пределами трапеции игнорируется;
- смена угла телефона меняет маску, но не ломает координаты;
- NaN/пустая depth map не роняет сервер;
- устаревшая depth map не используется;
- `STOP` немедленный, `GO` после трёх чистых результатов.

Ручные сцены:

1. пустой пол при ярком и слабом свете;
2. белая и цветная коробки в каждой полосе;
3. неизвестный объект, которого нет в YOLO prompts;
4. человек;
5. объект сбоку вне проезжей зоны;
6. поворот и изменение наклона телефона;
7. частично закрытая камера и полностью тёмный кадр.

Готово, когда:

- неизвестная коробка в центре стабильно даёт `STOP` минимум в 9 из 10 кадров;
- пустой пол не даёт ложный `STOP` дольше одного кадра;
- маршрут не дрожит между полосами;
- combined p95 не превышает примерно 400 мс на используемом Mac;
- отключение depth возвращает полностью рабочий YOLO-only режим;
- телефон и Mac показывают одинаковое решение и геометрию.

## 9. Godot — полезный второй этап

Godot не нужен для первого запуска модели, но очень полезен как повторяемый
стенд. Сделать сцену с плоскостью, камерой и кубами, менять положение/свет и
экспортировать одновременно RGB и настоящий depth buffer. Это позволит считать
ошибки lane occupancy и подобрать пороги без ручного передвижения коробок.

Не пытаться сначала идеально перенести физику реальной камеры в Godot. Первый
этап симулятора — только регрессионные сцены: empty, left, center, right,
blocked и outside.

## 10. Порядок коммитов

1. `chore: add depth anything smoke test`
2. `feat: add monocular depth inference`
3. `feat: derive lane occupancy from floor depth residuals`
4. `feat: fuse depth navigation with yolo labels`
5. `feat: add depth diagnostics to mac monitor`
6. `test: add depth navigation regression scenes`

После каждого этапа запускать Python compile/tests, JavaScript build-check и
`git diff --check`. Не добавлять в Git скачанные веса, кеш Hugging Face,
виртуальные окружения, depth previews и incident-файлы.
