import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# CSV файл
csv_path = 'C:/Users/Danya/moderatsiya-kartochek-5706/data/private_info/train.csv'
df = pd.read_csv(csv_path, delimiter='\t')

# Прогрузка столбцов
df.columns = df.columns.str.strip()

# переделка значений в столбце 'label_id' в строки
df['label_id'] = df['label_id'].astype(str)

# Разделение данных на тренировочные и валидационные наборы
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_id'])

# Параметры для генераторов изображений
batch_size = 32
img_height = 224
img_width = 224

# Папка с изображениями
base_dir = 'C:/Users/Danya/moderatsiya-kartochek-5706/data/train'

# Функция для динамического поиска изображения в подпапках
def path_from_subfolders(row, base_dir):
    for folder in ['cigs', 'other', 'pipes', 'roll_cigs', 'smoking']:
        potential_path = os.path.join(base_dir, folder, row['image_name'])
        if os.path.exists(potential_path):
            return potential_path
    return None  # Если изображение не найдено в папках, возвращаем None

#  функция к  DataFrame
train_df['file_path'] = train_df.apply(lambda row: path_from_subfolders(row, base_dir), axis=1)
val_df['file_path'] = val_df.apply(lambda row: path_from_subfolders(row, base_dir), axis=1)

# Проверка доступа к файлам
print("Первые строки DataFrame с путями к файлам:")
print(train_df.head())

# проверка доступа к изображениям
train_df = train_df.dropna(subset=['file_path'])
val_df = val_df.dropna(subset=['file_path'])

# генератор данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='file_path',  # Используем полный путь к файлу
    y_col='label_id',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='file_path',  # Используем полный путь к файлу
    y_col='label_id',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

#  количество классов из train_generator
num_classes = len(train_generator.class_indices)
print("Количество классов:", num_classes)

#  модель
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Ззаморозка слоёв базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Сохраение модели
model.save('cigarette_smoking_model.h5')

print("Обучение завершено и модель сохранена!")
