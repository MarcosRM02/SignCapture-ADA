# SignCapture ADA

Pipeline de adquisición y preparación de datos para el reconocimiento de lenguaje de signos americano (ASL). El módulo descarga datos crudos, extrae landmarks de manos mediante MediaPipe Hands y genera características finales listas para entrenamiento de modelos.

## Objetivo

El proyecto proporciona un pipeline ETL mantenible y reproducible estructurado en tres capas (Medallion Architecture):

- **Bronze**: datos crudos descargados y organizados sin transformaciones.
- **Silver**: extracción de landmarks (21 puntos por mano) mediante MediaPipe Hands. 
- **Gold**: normalización, feature engineering (ángulos entre falanges), aumentación de datos y división train/val/test.

Documentación ampliada:

- [docs/architecture.md](./docs/architecture.md)
- [docs/traceability.md](./docs/traceability.md)

## Puesta en marcha

1. Crear y activar el entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate 
```

2. Instalar dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Configurar variables de entorno:

Crea un archivo `.env` en la raíz del proyecto con:

```env
SIGNCAPTURE_ROOT=/ruta/absoluta/a/SignCapture
```

4. Verificar configuración:

Revisa y ajusta los parámetros en `config/settings.yaml` según tus necesidades:

```yaml
mediapipe:
  min_detection_confidence: 0.3
  max_num_hands: 1

general:
  seed: 42
  split_ratios:
    train: 0.6
    val: 0.2
    test: 0.2

dataset:
  num_instances_per_sign: 6800

augmentation:
  num_images_per_sign: 2200
  rotation_range: 45
  zoom_range: 0.1
  horizontal_flip: true
  num_augmentations: 10
```

## Ejecución del pipeline

### Pipeline completo

Ejecutar las tres capas en secuencia:

```bash
python -m pipelines.run_bronze
python -m pipelines.run_silver
python -m pipelines.run_gold
```

O usando el Makefile:

```bash
make bronze  # Capa Bronze: descarga de datos
make silver  # Capa Silver: extracción de landmarks
make gold    # Capa Gold: features finales
```

### Pipeline por capas

#### Bronze (Datos crudos)

Descarga el dataset desde Kaggle y lo organiza:

```bash
python -m pipelines.run_bronze
```

Resultado:
- `data/bronze/A/`, `data/bronze/B/`, ..., `data/bronze/Y/`
- 6,800 imágenes por letra (configurable)
- Formato: JPG, nombres normalizados (0001.jpg, 0002.jpg, ...)

#### Silver (Landmarks)

Extrae landmarks de manos usando MediaPipe Hands:

```bash
python -m pipelines.run_silver
```

Resultado:
- `data/silver/hand_landmarks.csv`
- 63 columnas (21 landmarks × 3 coordenadas: x, y, z)
- Metadatos: image_path, letter

#### Gold (Features finales)

Procesa landmarks y genera datasets finales:

```bash
python -m pipelines.run_gold
```

Resultado:
- `data/gold/train.csv`: conjunto de entrenamiento (con aumentación)
- `data/gold/val.csv`: conjunto de validación
- `data/gold/test.csv`: conjunto de test

## Flujo funcional

1. **Bronze**: El pipeline descarga imágenes desde Kaggle y las organiza por letra.
2. **Silver**: MediaPipe Hands extrae 21 landmarks (puntos clave) de cada mano detectada.
3. **Gold - Split**: Los datos se dividen estratificadamente en train/val/test (60/20/20).
4. **Gold - Augment**: Solo el set de entrenamiento se aumenta (rotación, zoom, flip).
5. **Gold - Normalize**: Los landmarks se normalizan a rango [-1, 1] por imagen.
6. **Gold - Features**: Se calculan 14 ángulos entre falanges y dedos adyacentes.

## Configuración

Las variables de entorno disponibles se documentan en `.env.example`:

- `SIGNCAPTURE_ROOT`: ruta raíz del proyecto que sigue la siguiente estructura:

```
SignCapture/
├── SignCapture-ADA/
├── SignCapture-ModelTests/
├── SignCapture-API/
├── SignCapture-Front/
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
└── models/
    ├── hand_landmarker.task
    └── ... (otros modelos futuros)
```

Los parámetros del pipeline se configuran en `config/settings.yaml`:

- `mediapipe.min_detection_confidence`: umbral de confianza para detección
- `mediapipe.max_num_hands`: número máximo de manos a detectar
- `general.seed`: semilla para reproducibilidad
- `general.split_ratios`: proporciones de división train/val/test
- `dataset.num_instances_per_sign`: imágenes por letra a descargar
- `augmentation.num_images_per_sign`: imágenes base para aumentación
- `augmentation.rotation_range`: rango de rotación en grados
- `augmentation.zoom_range`: rango de zoom (0.1 = ±10%)
- `augmentation.horizontal_flip`: activar flip horizontal
- `augmentation.num_augmentations`: aumentaciones por imagen

## Estructura de datos

### Bronze

Imágenes organizadas en carpetas por letra:

```
data/bronze/
├── A/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── B/
└── ...
```

### Silver

CSV con landmarks extraídos:

| image_path | letter | landmark0_x | landmark0_y | landmark0_z | ... |
|------------|--------|--------------|--------------|--------------|-----|
| bronze/A/0001.jpg | A | 0.456 | 0.789 | -0.012 | ... |

### Gold

CSV con features finales:

| original_id | letter | landmark0_x | ... | angle_thumb_1_2_3 | ... |angle_index_5_6_7 | ... |
|-------------|--------|--------------|-----|-------------|-------------|-----|-----|
| A_123 | A | -0.234 | ... | 45.6 | ... | 23.1 | ... |

Columnas adicionales:
- **original_id**: identificador de imagen original (rastreo de aumentación)
- **angle_thumb_...**, **angle_index_...**, **angle_middle_...**, **angle_ring_...**, **angle_pinky_...**: ángulos entre falanges
- **angle_thumb_index_...**, **angle_index_middle_...**, **angle_middle_ring_...**, **angle_ring_pinky_...**: ángulos entre dedos

## Utilidades

### Notebooks de análisis

El proyecto incluye notebooks Jupyter para análisis exploratorio:

- `notebooks/eda_bronze.ipynb`: análisis de imágenes crudas
- `notebooks/eda_silver.ipynb`: análisis de landmarks extraídos
- `notebooks/eda_gold.ipynb`: análisis de features finales

### Visualización de landmarks

Puedes visualizar landmarks sobre imágenes usando:

```python
from src.utils.visualization import annotate_image
from src.utils.io import load_image
from src.silver.landmark_detector import LandmarkDetector

detector = LandmarkDetector()
image = load_image("data/bronze/A/0001.jpg")
landmarks = detector.detect_landmarks(image)

if landmarks:
    annotated = annotate_image(image, landmarks)
    # Mostrar o guardar imagen anotada
```

## Notas

- El pipeline es idempotente: se puede ejecutar múltiples veces sin duplicar datos.
- La aumentación de datos se aplica solo al conjunto de entrenamiento.
- Los landmarks se normalizan independientemente por imagen para invarianza a escala y posición.
- El modelo MediaPipe Hands requiere el archivo `models/hand_landmarker.task` (descargado automáticamente).
- Las letras "J" y "Z" no están incluidas (requieren movimiento, fuera del alcance actual).
- La configuración usa un seed fijo (42) para garantizar reproducibilidad.

## Integración con otros módulos

Este módulo genera datos que son consumidos por:

- **SignCapture-ModelTests**: lee archivos de `data/gold/` para entrenar modelos
- **SignCapture-API**: debe replicar la normalización (`src/gold/normalizer.py`) para inferencia

**Importante**: Cualquier cambio en el proceso de normalización o feature engineering debe sincronizarse con SignCapture-API para garantizar consistencia entre entrenamiento e inferencia.
