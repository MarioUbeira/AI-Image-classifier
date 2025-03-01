import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image

tf.config.threading.set_intra_op_parallelism_threads(6)  # Ajusta según los núcleos de tu CPU (Yo tengo Intel Core i5-12400F 2.5 GHz con 6 núcleos)
tf.config.threading.set_inter_op_parallelism_threads(6)


Image.MAX_IMAGE_PIXELS = None 

print("Iniciando entrenamiento...")

# Directorio con imágenes clasificadas
datos_entrenamiento = './fotos_clasificadas/'

# Parámetros
altura, longitud = 150, 150
epocas = 20
batch_size = 32
lr = 0.0005

clases = len(os.listdir(datos_entrenamiento))
print(f"Se encontraron {clases} categorías en '{datos_entrenamiento}'.")

print("Preparando generadores de datos...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("Construyendo el modelo...")
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(altura, longitud, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(clases, activation='softmax')
])

print("Compilando el modelo...")
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=lr),
    metrics=['accuracy']
)

print("Iniciando entrenamiento...")
cnn.fit(
    train_generator,
    epochs=epocas,
    validation_data=val_generator,
)

print("Guardando modelo...")
if not os.path.exists('./modelo'):
    os.mkdir('./modelo')

cnn.save('./modelo/clasificador_fotos.h5')

print("¡Entrenamiento completado y modelo guardado!")
