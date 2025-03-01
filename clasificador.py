import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import shutil

print("Iniciando clasificación de imágenes...")

modelo_path = './modelo/clasificador_fotos.h5'
print(f"Cargando modelo desde: '{modelo_path}'...")
modelo = tf.keras.models.load_model(modelo_path)

carpeta_a_clasificar = './fotos_sin_clasificar/'
carpeta_origen = './fotos_clasificadas/'
carpeta_destino = './fotos_clasificadas_ia/'

categorias = sorted(os.listdir(carpeta_origen))
print(f"Categorías detectadas: {categorias}")

for categoria in categorias:
    os.makedirs(os.path.join(carpeta_destino, categoria), exist_ok=True)

imagenes_a_clasificar = [f for f in os.listdir(carpeta_a_clasificar) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
print(f"{len(imagenes_a_clasificar)} imágenes encontradas en '{carpeta_a_clasificar}'.")

for img_nombre in imagenes_a_clasificar:
    img_path = os.path.join(carpeta_a_clasificar, img_nombre)

    try:
        print(f"🔄 Clasificando '{img_nombre}'...")

        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

        prediccion = modelo.predict(img_array, verbose=0)[0]  # Obtener array sin logs extra
        categoria_predicha = categorias[np.argmax(prediccion)]

        destino = os.path.join(carpeta_destino, categoria_predicha, img_nombre)
        shutil.move(img_path, destino)
        print(f"'{img_nombre}' → '{categoria_predicha}'.")

    except (OSError, ValueError) as e:
        print(f"Error con '{img_nombre}': {e}")

print("¡Clasificación completa!")