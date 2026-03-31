# Trazabilidad y mantenimiento

## Responsabilidades por capa

### Bronze (Datos crudos)

Responsable de adquisicion y organizacion de datos sin transformaciones complejas. No debe contener logica de extraccion de features.

### Silver (Landmarks)

Responsable de transformar imagenes en representaciones vectoriales mediante deteccion de puntos clave. No debe contener logica de normalizacion o aumentacion.

### Gold (Features finales)

Responsable de preparar datos para modelado mediante normalizacion, feature engineering, aumentacion y division estratificada. No debe contener logica de entrenamiento de modelos.

### Utilidades

Responsables de funciones transversales de IO, visualizacion y estructuras de datos compartidas.

### Configuracion

Responsable de centralizar parametros y rutas del sistema. Debe ser el unico punto de acceso a variables de entorno y archivos YAML.

## Matriz de trazabilidad

| Necesidad funcional | Modulo principal | Elementos implicados |
| --- | --- | --- |
| Descargar dataset de Kaggle | `src/bronze/downloader.py` | `download_kaggle_dataset`, `kaggle.api` |
| Organizar imagenes por letra | `src/bronze/downloader.py` | `process_downloaded_dataset`, `_restructure_downloaded_dataset`, `PathsConfig` |
| Normalizar cantidad de imagenes | `src/bronze/downloader.py` | `_remove_images`, `DatasetConfig.num_instances_per_sign` |
| Detectar landmarks de mano | `src/silver/landmark_detector.py` | `LandmarkDetector.detect_landmarks`, `MediaPipe HandLandmarker` |
| Extraer landmarks de dataset | `src/silver/landmark_extractor.py` | `extract_landmarks`, `LandmarkDetector` |
| Dividir datos estratificadamente | `src/gold/splitter.py` | `split_landmarks`, `_split_by_original_id_balanced`, `_split_rows_balanced`, `GeneralConfig` |
| Aumentar datos de entrenamiento | `src/gold/augmentor.py` | `augment_landmarks`, `_apply_augmentations`, `_rotate_landmarks`, `_zoom_landmarks`, `_flip_landmarks_horizontal`, `AugmentationConfig` |
| Normalizar landmarks | `src/gold/normalizer.py` | `normalize_landmarks`, `LandmarkPoint`, escalado min-max por imagen |
| Calcular features angulares | `src/gold/feature_engineering.py` | `add_angle_features`, `_build_landmark_tensor`, `_compute_angle_between_three_points`, `_FALANGE_ANGLE_SPECS` |
| Visualizar landmarks | `src/utils/visualization.py` | `draw_landmarks_on_image`, `_draw_connections_on_image` |
| Cargar configuracion | `src/config.py` | `Config`, `PathsConfig`, `MediaPipeConfig`, `GeneralConfig`, `DatasetConfig`, `AugmentationConfig`, `load_yaml` |

## Reglas de mantenimiento

- Si cambia el formato de landmarks de MediaPipe, actualizar primero `src/utils/landmark.py` y `src/silver/landmark_detector.py`.
- Si cambia la estrategia de normalizacion, concentrar los cambios en `src/gold/normalizer.py` y sincronizar con SignCapture-API.
- Si cambian las features de angulos, mantener documentacion actualizada en `src/gold/feature_engineering.py`.
- Si cambian los ratios de split o aumentacion, actualizar `config/settings.yaml` sin modificar codigo.
- Si aparecen nuevas capas de datos, seguir la arquitectura Medallion (Bronze → Silver → Gold → ...).

## Convenciones de documentacion

- Cada modulo debe describir su responsabilidad en el docstring principal.
- Cada funcion publica debe explicar entradas, salidas y efectos secundarios.
- Las decisiones de arquitectura deben reflejarse en `docs/architecture.md`.
- Los parametros configurables deben reflejarse en `README.md` y `config/settings.yaml`.
- Los cambios que afectan a otros modulos (ModelTests, API) deben documentarse explicitamente.