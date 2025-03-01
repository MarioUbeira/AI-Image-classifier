import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import shutil

print("🔵 Iniciando clasificación de imágenes...")

# Cargar el modelo entrenado
modelo_path = './modelo/clasificador_fotos.h5'
print(f"Cargando modelo desde '{modelo_path}'...")
modelo = tf.keras.models.load_model(modelo_path)

carpeta_a_clasificar = './fotos_sin_clasificar/'  # Carpeta con imágenes sin clasificar
carpeta_origen = './fotos_clasificadas/'  # Carpeta donde están las categorías originales
carpeta_destino = './fotos_clasificadas_ia/'  # Nueva carpeta para guardar las clasificaciones de la IA

categorias = sorted(os.listdir(carpeta_origen))
print(f"Categorías detectadas: {categorias}")

if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)
    print(f"Creada carpeta '{carpeta_destino}'.")

for categoria in categorias:
    ruta_categoria = os.path.join(carpeta_destino, categoria)
    if not os.path.exists(ruta_categoria):
        os.makedirs(ruta_categoria)
        print(f"Creada subcarpeta '{ruta_categoria}'.")

imagenes_a_clasificar = os.listdir(carpeta_a_clasificar)
print(f"{len(imagenes_a_clasificar)} imágenes encontradas en '{carpeta_a_clasificar}'.")

for img_nombre in imagenes_a_clasificar:
    img_path = os.path.join(carpeta_a_clasificar, img_nombre)

    try:
        print(f"Clasificando '{img_nombre}'...")

        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

        # Hacer la predicción
        prediccion = modelo.predict(img_array)
        categoria_predicha = categorias[np.argmax(prediccion)]  # Categoría con mayor probabilidad

        destino = os.path.join(carpeta_destino, categoria_predicha, img_nombre)
        shutil.move(img_path, destino)
        print(f"'{img_nombre}' movida a '{categoria_predicha}'.")

    except Exception as e:
        print(f"Error con '{img_nombre}': {e}")

print("¡Clasificación completa!")
