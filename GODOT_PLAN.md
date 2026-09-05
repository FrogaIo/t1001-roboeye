# План симулятора RoboEye в Godot

Документ предназначен для следующего разработчика или AI-агента. Цель — сделать
не декоративную анимацию, а маленький воспроизводимый 3D-полигон, где робот
получает только RGB виртуальной камеры, отправляет его в существующий Python
RoboEye и объезжает препятствия по ответу компьютерного зрения.

## 1. Что увидит пользователь

После запуска открывается окно Godot в стиле операторского пульта:

```text
┌────────────────────────────────────────────────────────────────────┐
│ ROBOEYE SIM   ● VISION CONNECTED   ASSIST   GO   PING 84 ms       │
├───────────────────────────────────────────────┬────────────────────┤
│                                               │ ROBOT CAMERA       │
│     вид склада от третьего лица               │ RGB + boxes        │
│                                               │ route LEFT         │
│          [коробки]        [стол]              ├────────────────────┤
│                    🤖 ─ ─ ─ ➜                 │ LEFT  CENTER RIGHT │
│                                               │ clear busy   clear │
│   виртуальный джойстик                        │ YOLO 43 ms         │
├───────────────────────────────────────────────┴────────────────────┤
│ [MANUAL] [ASSIST] [AUTO]  [RESET] [RANDOMIZE]  collisions: 0      │
└────────────────────────────────────────────────────────────────────┘
```

Пользователь двигает джойстик, WASD или настоящий геймпад. В режиме `ASSIST`
команда пользователя остаётся основной, но зрение имеет право:

- запретить движение вперёд при `STOP`;
- разрешить задний ход;
- повернуть в сторону маршрута `LEFT` или `RIGHT`;
- продолжить движение после устойчивого `GO`.

При `STOP` экран кратко подсвечивается красным. Жёлтая линия на полу показывает
направление, которое выбрал vision controller. Справа виден **точно тот RGB**,
который ушёл в Python, с полученными обратно рамками и перспективной сеткой.

## 2. Главное правило честной симуляции

Контроллеру доступны только:

- RGB виртуальной monocamera;
- JSON-ответ RoboEye;
- команда джойстика;
- собственная скорость и поворот робота.

Нельзя использовать `RayCast3D`, координаты коробок, collision contacts или
внутренний depth buffer для выбора маршрута. Физические столкновения нужны для
реализма и метрик, но не для управления.

Godot ground-truth можно сохранить отдельно для тестов:

- фактическое столкновение;
- расстояния до объектов;
- истинно свободная полоса;
- позже — depth buffer через отдельный render pass.

Так можно доказать, что робот ехал по компьютерному зрению, а не по скрытым
данным игрового мира.

## 3. Версия и создание проекта

Использовать стабильный Godot 4.7.2 Standard, не .NET. Создать проект в
подкаталоге репозитория:

```text
t1001-roboeye/
├── app.py
├── detector.py
├── ...
└── godot/
    ├── project.godot
    ├── scenes/
    ├── scripts/
    ├── materials/
    └── assets/
```

После появления проекта добавить в `.gitignore`:

```gitignore
godot/.godot/
godot/export/
godot/screenshots/
```

Проверить запуск пустого проекта:

```bash
godot --editor --path godot
```

Если CLI не установлен, открыть Godot.app и импортировать `godot/project.godot`.

## 4. Структура сцен

### `main.tscn`

```text
Main (Node3D)
├── WorldEnvironment
├── DirectionalLight3D
├── Floor (StaticBody3D)
│   ├── MeshInstance3D
│   └── CollisionShape3D
├── Walls (Node3D)
├── Obstacles (Node3D)
├── Robot (instance robot.tscn)
├── OperatorCamera (Camera3D)
├── VisionViewport (SubViewport, 640×360, UPDATE_ALWAYS)
│   └── VisionCamera (Camera3D)
├── VisionClient (Node)
├── ScenarioManager (Node)
├── Metrics (Node)
└── HUD (CanvasLayer, instance hud.tscn)
```

`VisionCamera` копирует `global_transform` маркера `Robot/CameraAnchor`. Камера
должна быть дочерней `VisionViewport`, иначе viewport не получит её изображение.
`OperatorCamera` остаётся отдельной и показывает сцену от третьего лица.

### `robot.tscn`

```text
Robot (CharacterBody3D)
├── CollisionShape3D
├── BodyMesh (MeshInstance3D)
├── LeftWheel / RightWheel (MeshInstance3D)
├── CameraAnchor (Marker3D, высота 0.55 м, наклон 12–18° вниз)
└── CollisionArea (Area3D, только для metrics)
```

Для MVP использовать `CharacterBody3D`, а не автомобильную физику. Это
дифференциальный робот: линейная скорость вперёд и угловая скорость вокруг Y.
Такой корпус проще, устойчивее и предсказуемее для агента.

### `hud.tscn`

- статус соединения;
- режим `MANUAL / ASSIST / AUTO`;
- `GO / STOP` и reason;
- `PING`, YOLO ms, позже depth ms;
- миниатюра `VisionViewportTexture`;
- canvas overlay рамок, сетки и маршрута;
- три индикатора полос;
- виртуальный joystick;
- `RESET` и `RANDOMIZE`;
- счётчики времени, столкновений и пройденной дистанции.

## 5. Полигон

Сначала не импортировать тяжёлые ассеты. Собрать всё примитивами Godot:

- пол 14×20 м с неброской текстурой;
- стены и несколько колонн;
- белая коробка;
- яркие синие/красные брендированные кубы;
- стол из BoxMesh;
- стул из нескольких BoxMesh;
- неизвестное препятствие необычной формы;
- CapsuleMesh как временный человек.

У каждого препятствия должен быть `StaticBody3D + CollisionShape3D`. Материалы,
цвет и освещение рандомизируются кнопкой `RANDOMIZE`, но размеры и collision
shape остаются валидными.

Готовые сценарии:

1. `empty` — свободный пол;
2. `center_box` — коробка по центру;
3. `left_right_choice` — центр и одна боковая полоса заняты;
4. `blocked` — заняты все варианты;
5. `unknown_object` — объект без YOLO-класса;
6. `low_light` — слабое освещение;
7. `colored_cubes` — сцена, похожая на реальные кубы.

## 6. Передача изображения в Python

Для симулятора запустить текущий backend без TLS, только на loopback:

```bash
cd /Users/roman/Documents/GitHub/t1001-roboeye
source venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8000
```

Godot подключается к:

```text
ws://127.0.0.1:8000/ws
```

Поток уже совместим с текущим протоколом RoboEye:

1. text JSON с углом виртуальной камеры;
2. binary JPEG;
3. text JSON-ответ с detections, lanes, state, route и corridor.

Каркас `scripts/vision_client.gd`:

```gdscript
class_name VisionClient
extends Node

signal result_received(result: Dictionary)
signal connection_changed(connected: bool)

@export var url := "ws://127.0.0.1:8000/ws"
var socket := WebSocketPeer.new()
var waiting_for_result := false

func _ready() -> void:
    socket.connect_to_url(url)

func _process(_delta: float) -> void:
    socket.poll()
    var open := socket.get_ready_state() == WebSocketPeer.STATE_OPEN
    connection_changed.emit(open)

    if not open:
        return

    while socket.get_available_packet_count() > 0:
        var packet := socket.get_packet()
        if socket.was_string_packet():
            var parsed = JSON.parse_string(packet.get_string_from_utf8())
            if parsed is Dictionary:
                waiting_for_result = false
                result_received.emit(parsed)

func send_camera_frame(image: Image, pitch_degrees: float) -> void:
    if socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
        return
    if waiting_for_result:
        return

    socket.send_text(JSON.stringify({
        "type": "orientation",
        "pitch_degrees": pitch_degrees,
    }))
    socket.send(image.save_jpg_to_buffer(0.68), WebSocketPeer.WRITE_MODE_BINARY)
    waiting_for_result = true
```

`VisionCapture` запускается таймером раз в `0.333` секунды:

```gdscript
func capture_and_send() -> void:
    if vision_client.waiting_for_result:
        return
    await RenderingServer.frame_post_draw
    var image := vision_viewport.get_texture().get_image()
    vision_client.send_camera_frame(image, abs(vision_camera.rotation_degrees.x))
```

Не читать viewport в `_ready()`: изображение может быть чёрным или устаревшим.
Не отправлять новый кадр, пока не пришёл ответ, иначе накопится latency.

## 7. Управление роботом

Создать Input Map:

```text
move_forward:  W, left stick up
move_back:     S, left stick down
turn_left:     A, left stick left
turn_right:    D, left stick right
mode_manual:   1
mode_assist:   2
mode_auto:     3
reset:         R
```

Ввод:

```gdscript
var stick := Input.get_vector(
    "turn_left", "turn_right", "move_forward", "move_back"
)
var requested_turn := stick.x
var requested_throttle := -stick.y
```

Физика:

```gdscript
rotation.y += final_turn * turn_speed * delta
velocity = -global_transform.basis.z * final_throttle * max_speed
move_and_slide()
```

Не задавать `global_position` напрямую: это обходит нормальную collision physics.

## 8. Safety controller

Хранить время последнего результата. Если ответ старше `1.0` секунды или связь
пропала, движение вперёд запрещается.

### MANUAL

Джойстик полностью управляет роботом. Vision только рисуется и логируется.

### ASSIST — основной демонстрационный режим

```text
GO:
  throttle = joystick throttle
  turn = joystick turn

STOP + LEFT:
  forward throttle = 0
  reverse разрешён
  turn = -0.75, пока центр не станет свободным

STOP + RIGHT:
  forward throttle = 0
  reverse разрешён
  turn = +0.75, пока центр не станет свободным

STOP + BLOCKED:
  throttle = 0
  turn = 0
```

Зафиксировать выбранную сторону минимум на 1 секунду, чтобы робот не дёргался
между `LEFT` и `RIGHT`. После трёх `GO` текущий `DecisionFilter` разрешит ехать.

### AUTO — после работающего ASSIST

Состояния:

```text
CRUISE → BRAKE → COMMIT_TURN → BYPASS → RECENTER → CRUISE
```

- `CRUISE`: ехать вперёд с 55% скорости;
- `BRAKE`: остановиться при `STOP`;
- `COMMIT_TURN`: повернуть по route до устойчивого `GO`;
- `BYPASS`: проехать вперёд фиксированное время или расстояние;
- `RECENTER`: плавно вернуться к сохранённому исходному heading;
- `BLOCKED`: стоять и ждать или медленно сдавать назад.

AUTO не должен быть первым этапом. Если ASSIST не объезжает три простые сцены,
полный автомат только спрячет ошибку за дополнительным state machine.

## 9. Отрисовка vision overlay в Godot

Python уже возвращает нормализованные координаты. HUD должен рисовать:

- `detection.box` поверх миниатюры камеры;
- `corridor` как три перспективных полигона;
- красный занятый и зелёный свободный сектор;
- жёлтую стрелку route;
- красную рамку экрана при `STOP`.

Эту математику нужно перенести из `static/app.js`, сохранив нормализованные
координаты. Нельзя рисовать новую «примерную» сетку, не совпадающую с сервером.

В 3D-виде от третьего лица можно показать жёлтые точки будущей траектории, но
они должны строиться только из текущего route и heading робота. Это визуализация,
не дополнительный сенсор.

## 10. Метрики

`metrics.gd` записывает JSONL или CSV:

```text
timestamp, scenario, mode, vision_state, route,
speed, collision_count, distance_travelled, stale_ms
```

Минимальные показатели для демо:

- число столкновений;
- время прохождения сценария;
- количество STOP;
- доля времени с устаревшим vision response;
- средний/p95 round trip;
- выбранный маршрут против ground-truth свободной полосы.

Логи симуляции хранить в `godot/logs/` и игнорировать в Git.

## 11. Реалистичный порядок реализации

### Этап A — 30–45 минут: мир и ручное движение

- создать проект и `main.tscn`;
- пол, свет, стены и три коробки;
- `CharacterBody3D`-робот;
- WASD/gamepad;
- operator camera и reset.

Результат: робот ездит и физически сталкивается с коробками.

### Этап B — 30–45 минут: виртуальная камера и WebSocket

- `SubViewport 640×360`;
- копирование transform камеры;
- JPEG 3 Гц;
- `VisionClient`;
- JSON status в HUD.

Результат: `/monitor` на Mac показывает картинку именно из Godot, YOLO видит
коробки, а Godot получает `GO/STOP`.

### Этап C — 30 минут: ASSIST

- safety timeout;
- запрет движения вперёд при STOP;
- committed LEFT/RIGHT turn;
- красная сигнализация;
- три контрольных сценария.

Результат: пользователь держит джойстик вперёд, а робот сам тормозит и
поворачивает вокруг центральной коробки.

### Этап D — 30–60 минут: красивый HUD

- виртуальный joystick;
- robot-camera inset;
- рамки и перспективная сетка;
- mode/reset/randomize;
- метрики.

### Этап E — после интеграции Depth Anything

- неизвестные препятствия;
- fusion depth + YOLO;
- depth overlay только в диагностическом режиме;
- сохранение RGB + ground truth для регрессионных тестов.

## 12. Критерии приёмки MVP

- Godot стартует без ручной настройки путей;
- Python запускается одной командой;
- virtual camera стабильно приходит в `/monitor`;
- после прогрева round trip не накапливается;
- отключение Python останавливает движение вперёд максимум за 1 секунду;
- режим MANUAL не маскируется автоматикой;
- режим ASSIST объезжает центральную коробку влево и вправо;
- `BLOCKED` полностью останавливает робота;
- физические collision data не участвуют в выборе steering;
- reset возвращает робота и сценарий в исходное состояние;
- все generated/cache/log файлы исключены из Git.

## 13. Что не делать в первой версии

- не использовать `VehicleBody3D` и реалистичную модель шин;
- не строить SLAM;
- не добавлять ROS;
- не делать сетевой джойстик с телефона до работающего локального ASSIST;
- не импортировать огромный warehouse asset pack;
- не запускать YOLO внутри Godot;
- не передавать Godot координаты объектов из сцены;
- не смешивать depth ground truth и monocular depth модели.

## 14. Коммиты

1. `feat(godot): scaffold simulation world`
2. `feat(godot): add differential robot controls`
3. `feat(godot): stream virtual camera to roboeye`
4. `feat(godot): add vision assisted driving`
5. `feat(godot): render operator hud and camera overlay`
6. `test(godot): add repeatable obstacle scenarios`
7. `feat(godot): add autonomous bypass state machine`

После каждого коммита запускать проект из чистого Godot editor, проверять
Python compile/tests и `git diff --check`. Не включать `camera_test.py` без
явного решения владельца файла.
