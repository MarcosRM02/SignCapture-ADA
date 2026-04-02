# Arquitectura

## Visión general

El módulo SignCapture-ADA implementa un pipeline ETL basado en la arquitectura Medallion (Bronze → Silver → Gold). Cada capa transforma los datos progresivamente desde su estado crudo hasta features listas para modelado, manteniendo separación de responsabilidades y trazabilidad completa.

## Objetivos de diseño

- Mantener el pipeline modular y cada capa independiente.
- Facilitar la reproducibilidad mediante configuración centralizada y seed fijo.
- Hacer trazable el linaje de datos desde descarga hasta features finales.
- Encapsular dependencias externas como MediaPipe y Kaggle API.

## Estructura de capas

### Capa Bronze (Datos Crudos)

**Responsabilidad**: Adquisición y organización de datos sin transformaciones complejas.

**Módulo**: `src/bronze/downloader.py`

**Flujo**:
1. Descarga dataset desde Kaggle usando API oficial.
2. Descomprime y organiza imágenes por letra (A-Y, excluyendo J y Z).
3. Elimina variantes con rotaciones explícitas en el nombre.
4. Normaliza cantidad de imágenes por letra según configuración.
5. Renombra archivos a formato secuencial (0001.jpg, 0002.jpg, ...).

**Salida**: Directorio `data/bronze/` con carpetas por letra, imágenes JPG.

**Decisiones técnicas**:
- Se excluyen J y Z porque requieren movimiento (fuera del alcance de clasificación estática).
- La normalización de cantidad evita desbalanceo entre clases.
- El renombrado facilita trazabilidad y debugging.

### Capa Silver (Extracción de Landmarks)

**Responsabilidad**: Transformar imágenes en representaciones vectoriales mediante detección de puntos clave.

**Módulos**:
- `src/silver/landmark_detector.py`: Envoltorio sobre MediaPipe HandLandmarker.
- `src/silver/landmark_extractor.py`: Procesamiento batch de imágenes.

**Flujo**:
1. Lee imágenes de Bronze capa por capa.
2. Detecta mano dominante (máximo 1 mano por configuración).
3. Extrae 21 landmarks con coordenadas (x, y, z) normalizadas por MediaPipe.
4. Guarda landmarks en CSV con metadatos (image_path, letter).

**Salida**: Archivo `data/silver/hand_landmarks.csv` con 63 columnas numéricas.

**Decisiones técnicas**:
- MediaPipe provee landmarks ya normalizados (x, y en [0, 1], z relativo al plano).
- Se usa CSV para facilitar inspección y compatibilidad con herramientas estándar.
- La detección de confianza mínima (0.3) filtra imágenes con manos poco visibles.

### Capa Gold (Features Finales)

**Responsabilidad**: Preparar datos para entrenamiento mediante normalización, feature engineering, aumentación y división estratificada.

**Módulos**:
1. `src/gold/splitter.py`: División estratificada en train/val/test.
2. `src/gold/augmentor.py`: Aumentación de datos solo en training set.
3. `src/gold/normalizer.py`: Normalización de landmarks a rango [-1, 1].
4. `src/gold/feature_engineering.py`: Cálculo de ángulos entre falanges.

**Flujo**:
1. **Split**: Divide Silver en train/val/test (60/20/20) manteniendo balance exacto.
2. **Augment**: Genera 10 variantes por muestra de train (rotación, zoom, flip).
3. **Normalize**: Escala landmarks a [-1, 1] por imagen para invarianza.
4. **Features**: Calcula 14 ángulos adicionales entre puntos de referencia.
5. **Export**: Guarda tres archivos CSV (train, val, test) con 77 columnas.

**Salida**: Directorio `data/gold/` con train.csv, val.csv, test.csv.

**Decisiones técnicas**:
- La aumentación solo en train evita data leakage hacia validación/test.
- La normalización por imagen permite invarianza a escala y traslación.
- Los ángulos capturan relaciones geométricas entre dedos (features interpretables).
- CSV ofrece simplicidad en la inspección manual y compatibilidad con herramientas simples.

## Estructura de carpetas

```
SignCapture-ADA/
├── src/
│   ├── bronze/
│   │   ├── __init__.py
│   │   └── downloader.py          # Descarga y organización de datos crudos
│   ├── silver/
│   │   ├── __init__.py
│   │   ├── landmark_detector.py   # Envoltorio de MediaPipe
│   │   └── landmark_extractor.py  # Extracción batch de landmarks
│   ├── gold/
│   │   ├── __init__.py
│   │   ├── splitter.py            # División estratificada
│   │   ├── augmentor.py           # Aumentación de datos
│   │   ├── normalizer.py          # Normalización de landmarks
│   │   └── feature_engineering.py # Cálculo de ángulos
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py                  # Lectura/escritura de archivos
│   │   ├── landmark.py            # Dataclass LandmarkPoint
│   │   └── visualization.py       # Renderizado de landmarks
│   └── config.py                  # Configuración centralizada
├── pipelines/
│   ├── run_bronze.py              # Pipeline Bronze
│   ├── run_silver.py              # Pipeline Silver
│   └── run_gold.py                # Pipeline Gold
├── config/
│   └── settings.yaml              # Configuración YAML
├── notebooks/                     # Análisis exploratorio
│   ├── eda_bronze.ipynb
│   ├── eda_silver.ipynb
│   └── eda_gold.ipynb
├── tests/                         # Pruebas unitarias
├── requirements.txt               # Dependencias Python
└── Makefile                       # Comandos de automatización
```

## Flujo de datos completo

```
┌─────────────────────────────────────────────────────────────┐
│ BRONZE LAYER                                                │
│                                                             │
│ Kaggle API                                                  │
│     ↓                                                       │
│ Descarga ZIP                                                │
│     ↓                                                       │
│ Extracción y filtrado                                       │
│     ↓                                                       │
│ Normalización de cantidad                                   │
│     ↓                                                       │
│ Renombrado secuencial                                       │
│     ↓                                                       │
│ Output: data/bronze/{A-Y}/*.jpg                            │
│         (6,800 imágenes/letra)                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ SILVER LAYER                                                │
│                                                             │
│ Lectura de imágenes JPG                                     │
│     ↓                                                       │
│ MediaPipe HandLandmarker                                    │
│     ↓                                                       │
│ Extracción de 21 landmarks × 3 coords                       │
│     ↓                                                       │
│ Validación de confianza (>0.3)                             │
│     ↓                                                       │
│ Output: data/silver/hand_landmarks.csv                     │
│         (63 columnas numéricas + metadatos)                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ GOLD LAYER                                                  │
│                                                             │
│ 1. SPLIT (splitter.py)                                     │
│    ├─> Train (60%)                                         │
│    ├─> Val (20%)                                           │
│    └─> Test (20%)                                          │
│                                                             │
│ 2. AUGMENT (augmentor.py - solo train)                    │
│    ├─> Rotación ±45°                                      │
│    ├─> Zoom ±10%                                          │
│    ├─> Flip horizontal                                     │
│    └─> 10 aumentaciones/muestra                           │
│                                                             │
│ 3. NORMALIZE (normalizer.py - todos los sets)             │
│    └─> Landmarks escalados a [-1, 1] por imagen           │
│                                                             │
│ 4. FEATURES (feature_engineering.py - todos los sets)     │
│    ├─> 10 ángulos entre falanges                          │
│    └─> 4 ángulos entre dedos adyacentes                   │
│                                                             │
│ Output: data/gold/{train,val,test}.csv                 │
│         (77 columnas: 63 landmarks + 14 ángulos)           │
└─────────────────────────────────────────────────────────────┘
```

## Configuración centralizada

### `src/config.py`

Define 5 dataclasses para configuración:

- **PathsConfig**: Rutas de datos (root, bronze, silver, gold).
- **MediaPipeConfig**: Parámetros de detección (confianza, num_manos, modelo).
- **GeneralConfig**: Seed de reproducibilidad, ratios de split.
- **DatasetConfig**: Repositorio de Kaggle, instancias por signo.
- **AugmentationConfig**: Parámetros de aumentación (rotación, zoom, flip, cantidad).

La instancia global `config = Config()` carga desde `config/settings.yaml` con fallback a variables de entorno.

### `config/settings.yaml`

Parámetros clave:

```yaml
mediapipe:
  min_detection_confidence: 0.3  # Balance entre cobertura y precisión
  max_num_hands: 1               # Solo mano dominante

general:
  seed: 42                       # Reproducibilidad
  split_ratios:
    train: 0.6
    val: 0.2
    test: 0.2

dataset:
  num_instances_per_sign: 6800   # Balance de clases

augmentation:
  num_images_per_sign: 2200      # Base para aumentación
  num_augmentations: 10          # Multiplicador de dataset
```

## Decisiones técnicas clave

### MediaPipe vs soluciones alternativas

- **Ventaja**: Modelos preentrenados de Google, alta precisión, sin entrenamiento.
- **Desventaja**: Dependencia externa, difícil de personalizar.
- **Justificación**: Para prototipado rápido y dataset estándar, MediaPipe ofrece la mejor relación velocidad/precisión.

### Normalización por imagen

- **Alternativa**: Normalización global usando estadísticas del dataset completo.
- **Elección**: Normalización independiente por imagen.
- **Justificación**: Invarianza a escala y posición de la mano en el frame, robustez ante diferentes distancias de cámara.

### Aumentación solo en train

- **Riesgo**: Data leakage si se aplica a val/test antes del split.
- **Estrategia**: Aumentación después del split, exclusivamente en train.
- **Justificación**: Garantiza que val/test evalúen generalización real, no memorización de transformaciones.

## Puntos de extensión

1. **Nuevas fuentes de datos**: Añadir otros datasets manteniendo formato Bronze compatible.
2. **Features adicionales**: Extender `feature_engineering.py` con velocidades, aceleraciones (para video).
3. **Validación de calidad**: Integrar Great Expectations o similar para validación de esquema.
4. **Tracking de experimentos**: Integrar MLflow para versionar datos y parámetros.
5. **Procesamiento distribuido**: Migrar a Spark o Dask para datasets >100M muestras.

## Riesgos actuales

1. **Escalabilidad**: Pipeline secuencial, no optimizado para datasets masivos (>1M imágenes).
2. **Falta de versionado**: No hay control de versiones de datos (DVC, Delta Lake).
3. **Validación limitada**: No hay tests exhaustivos de calidad de datos entre capas.
4. **MediaPipe blackbox**: Difícil debuggear problemas de detección internos del modelo.

## Mantenibilidad

- **Idempotencia**: Todos los pipelines son reejecutabless sin efectos secundarios.
- **Configuración externa**: Cambios en parámetros sin modificar código.
- **Modularidad**: Cada capa puede evolucionar independientemente.
- **Trazabilidad**: `original_id` rastrea muestras desde Bronze hasta Gold.
- **Reproducibilidad**: Seed fijo garantiza resultados deterministas.
