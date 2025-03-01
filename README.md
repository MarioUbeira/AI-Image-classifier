# Clasificador de Imágenes con IA

Este proyecto utiliza una red neuronal convolucional (CNN) en TensorFlow para clasificar automáticamente imágenes en distintas categorías. Se entrena con imágenes previamente clasificadas por el usuario y luego se usa para clasificar nuevas imágenes de manera automática.

## Características
- Entrena un modelo de reconocimiento de imágenes basado en carpetas de clasificación existentes.
- Clasifica nuevas imágenes y las mueve a carpetas correspondientes.
- Mantiene separadas las imágenes clasificadas manualmente y las clasificadas por la IA.
- Muestra mensajes en la terminal para informar el progreso del entrenamiento y la clasificación.

## Requisitos
### Hardware
- CPU: Intel Core i5 o superior (soporta optimización en CPU)
- GPU: (Opcional) Se recomienda NVIDIA con CUDA para aceleración con TensorFlow
- Memoria RAM: 8GB o más

### Software
- Python 3.8 o superior
- TensorFlow
- Keras
- NumPy
- Pillow
- shutil

## Instalación
1. **Clonar el repositorio:**
   ```sh
   git clone https://github.com/tu_usuario/ia-clasificador-imagenes.git
   cd ia-clasificador-imagenes
   ```

2. **Crear un entorno virtual y activarlo:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate  # En Windows
   ```

3. **Instalar dependencias:**
   ```sh
   pip install -r requirements.txt
   ```

## Uso
### 1. Preparar las carpetas
   - `fotos_clasificadas/`: Contiene subcarpetas con imágenes ya clasificadas (por ejemplo, "familia", "amigos", "mascotas").
   - `fotos_sin_clasificar/`: Carpeta con imágenes que se quieren clasificar automáticamente.
   - `fotos_clasificadas_ia/`: Se crearán subcarpetas aquí con las imágenes clasificadas por la IA.

### 2. Entrenar el modelo
   ```sh
   python entrenar.py
   ```
   - Detecta las categorías en `fotos_clasificadas/`.
   - Entrena el modelo con las imágenes disponibles.
   - Guarda el modelo entrenado en la carpeta `modelo/`.

### 3. Clasificar nuevas imágenes
   ```sh
   python clasificador.py
   ```
   - Carga el modelo entrenado.
   - Clasifica las imágenes en `fotos_sin_clasificar/`.
   - Mueve las imágenes clasificadas a `fotos_clasificadas_ia/`.

## Mejoras y optimizaciones
- Uso de `batch_size` configurable para acelerar el entrenamiento.
- Posibilidad de implementar `mixed_precision` para optimizar en CPU/GPU.
- Uso de Google Colab para entrenamiento en GPU si no se dispone de una compatible.