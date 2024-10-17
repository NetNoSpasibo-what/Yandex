import cv2
import numpy as np

# Загрузка модели YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Загрузка классов объектов
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Загрузка видео
cap = cv2.VideoCapture('input_video.mp4')

# Получение параметров видео
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Создание объекта VideoWriter для записи результата
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    # Считывание кадра из видео
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Преобразование кадра в blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Задание входа и выполнение прямого прохода через сеть YOLO
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Инициализация списков для обнаруженных объектов, их координат и их вероятностей
    boxes = []
    confidences = []
    class_ids = []

    # Проход по каждому выходному слою
    for output in layer_outputs:
        # Проход по каждому обнаруженному объекту на выходном слое
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 16:  # Класс "dog"
                # Получение координат объекта на изображении
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Сохранение координат, вероятности и класса объекта
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применение non-maximum suppression для устранения дубликатов
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение результатов на кадре
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 255, 0)  # Зеленый цвет для всех прямоугольников
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "MANGO: {:.2f}".format(confidences[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 1, color, 2)

    # Запись кадра с обнаруженными объектами в выходной видео файл
    out.write(frame)

# Освобождение ресурсов
cap.release()
out.release()
