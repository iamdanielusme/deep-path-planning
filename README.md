# Generación de Trayectorias en Mapas con Obstáculos mediante Aprendizaje Profundo

**Universidad de Antioquia — Departamento de Ingeniería Electrónica**  
**Curso: Deep Learning | Proyecto Final — 2026**

---

## Equipo

| Nombre | Correo institucional | Rol en el proyecto |
|---|---|---|
| Angee Lorena Ocampo | angee.ocampo@udea.edu.co | Generación del dataset · GAN |
| Daniela Escobar | daniela.escobarv@udea.edu.co | CNN |
| Daniel Usme Urrea | dalejandro.usme@udea.edu.co | U-Net |

---

## Descripción del problema

La planificación de trayectorias es un problema fundamental en robótica y sistemas autónomos. En su formulación clásica, algoritmos como A\* resuelven el problema de forma exacta pero requieren ejecutarse en tiempo real para cada nueva consulta.

Este proyecto explora si una red neuronal profunda puede **aprender la política implícita** de un planificador óptimo: dado un mapa 2D con obstáculos, un punto de inicio y un punto de meta, el modelo debe predecir la trayectoria que los conecta sin invocar ningún algoritmo de búsqueda durante la inferencia.

El problema se formula como **segmentación semántica binaria pixel a pixel**: la red recibe el mapa codificado como una imagen de 3 canales y produce una máscara donde los píxeles activos representan la ruta generada.

---

## Estructura del repositorio

```
deep-path-planning/
│
├── dataset/                        # Archivos .npz generados (ver instrucciones abajo)
├── notebooks/
│   ├── 01_dataset_generation.ipynb # Pipeline de generación con A* (Angee Ocampo)
│   ├── 02_unet.ipynb               # Arquitectura U-Net con ablaciones (Daniel Usme)
│   ├── 03_cnn.ipynb                # Arquitectura CNN (Daniela Escobar)
│   └── 04_gan.ipynb                # Arquitectura GAN generativa (Angee Ocampo)
├── figures/                        # Curvas de entrenamiento y visualizaciones
├── requirements.txt
└── README.md
```

---

## Dataset

| Atributo | Detalle |
|---|---|
| **Origen** | Sintético — generado con el algoritmo A\* sobre mapas aleatorios 20×20 |
| **Tamaño** | 5.000 ejemplos |
| **Partición** | 80 % entrenamiento / 20 % test (`random_state=42`) |
| **Entrada X** | Tensor `(N, 20, 20, 3)` — canal 1: obstáculos, canal 2: inicio, canal 3: meta |
| **Salida Y** | Tensor `(N, 20, 20, 1)` — máscara binaria de la trayectoria óptima |

Los mapas se generan con densidad de obstáculos del 20 % (distribución Bernoulli). Las trayectorias se remuestrean a longitud fija de 40 puntos y se convierten a máscara binaria antes de guardarse en formato `.npz`.

> El dataset no se incluye en el repositorio. Para regenerarlo, ejecutar `01_dataset_generation.ipynb` en su totalidad.

### Consideraciones éticas del dataset

El dataset es completamente sintético, por lo que no involucra datos personales ni riesgos de privacidad. Sin embargo, al usar A\* con heurística Manhattan como oráculo, el modelo hereda el sesgo de esta métrica hacia trayectorias rectilíneas, lo que puede limitar la generalización a entornos con movimiento diagonal o costos variables.

---

## Metodología

Los tres modelos comparten la misma representación de entrada y salida, lo que garantiza una **comparación controlada y justa** entre arquitecturas.

### U-Net — Daniel Usme Urrea

Red encoder-decoder con conexiones de salto (*skip connections*) simétricas entre los niveles del codificador y del decodificador. Esta estructura permite que el modelo combine representaciones de contexto global (cuello de botella) con detalle espacial fino (capas superficiales), lo cual es especialmente adecuado para la predicción píxel a píxel de trayectorias continuas.

- Profundidad configurable: 2, 3 o 4 niveles de encoder/decoder
- Deconvolución aprendible con `Conv2DTranspose`
- `BatchNormalization` en cada bloque convolucional
- `Dropout` en el cuello de botella para regularización
- Padding interno para garantizar compatibilidad dimensional con cualquier profundidad

### CNN — Daniela Escobar

*(completar)*

### GAN — Angee Lorena Ocampo

*(completar)*

---

## Función de pérdida

Dado el fuerte desbalance de clases (la trayectoria ocupa menos del 10 % de los píxeles del mapa), se utiliza una pérdida combinada:

$$\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$

La **Binary Cross-Entropy** penaliza errores píxel a píxel, mientras que el **Dice Loss** maximiza el solapamiento global entre la predicción y la máscara real, siendo más robusto al desbalance que la entropía cruzada pura.

---

## Métricas de evaluación

| Métrica | Descripción |
|---|---|
| **Dice Coefficient** | Solapamiento entre máscara predicha y real. Robusto al desbalance de clases |
| **IoU** | Intersección sobre la unión. Estándar en segmentación semántica |
| **Tasa de conectividad** | Porcentaje de predicciones donde la ruta conecta inicio y meta sin interrupciones (verificación por BFS) |
| **Accuracy** | Reportada como referencia base; poco informativa por el desbalance |

La **tasa de conectividad** es la métrica más relevante para el dominio de aplicación: una trayectoria fragmentada es inutilizable en la práctica, independientemente de su Dice score.

---

## Ablaciones (U-Net)

Se realizó una búsqueda por ablación controlada —variando un único hiperparámetro respecto a un baseline fijo— sobre un subconjunto de 1.500 ejemplos para reducir el costo computacional. Los hiperparámetros explorados fueron:

| Hiperparámetro | Valores explorados | Justificación |
|---|---|---|
| `depth` | 2, **3**, 4 | Campo receptivo y capacidad de abstracción |
| `base_filters` | 16, **32**, 64 | Capacidad representacional del modelo |
| `dropout` | 0.0, **0.1**, 0.3 | Nivel de regularización |
| `use_transpose` | **True**, False | `Conv2DTranspose` vs `UpSampling2D` |
| `use_batchnorm` | **True**, False | Estabilidad del entrenamiento |

Los valores en negrita corresponden al baseline. El modelo final se reentrenó con la mejor configuración sobre el dataset completo (máximo 60 épocas, `EarlyStopping` con `patience=8` sobre `val_dice_coef`, `ReduceLROnPlateau` con `factor=0.5`).

---

## Reproducibilidad

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Generación del dataset

Ejecutar `notebooks/01_dataset_generation.ipynb` en su totalidad. Los archivos `.npz` se guardarán en `dataset/`.

### Entrenamiento

Ejecutar cada notebook de arquitectura de forma independiente (`02`, `03`, `04`). Los pesos del mejor modelo se guardan automáticamente (`best_unet.keras`, etc.) mediante `ModelCheckpoint`.

### Carga de un modelo entrenado

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "best_unet.keras",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef":     dice_coef,
        "iou_coef":      iou_coef
    }
)
```

### Semillas aleatorias

Todos los experimentos utilizan `random_state=42` (scikit-learn) y `np.random.default_rng(42)` (NumPy) para garantizar la reproducibilidad de particiones y submuestreos.

---

## Consideraciones éticas

**Justificación del uso de Deep Learning:** el problema de planificación de trayectorias cuenta con soluciones clásicas exactas y eficientes. El uso de redes neuronales se justifica como exploración del aprendizaje de políticas implícitas con potencial aplicación en escenarios donde la velocidad de inferencia es crítica (robótica en tiempo real, videojuegos, sistemas embebidos) y se puede tolerar una solución aproximada a cambio de latencia reducida.

**Impacto de errores:** una trayectoria predicha con cortes llevaría a un agente autónomo a un estado sin salida o a una colisión. Por este motivo, la tasa de conectividad se reporta como métrica de seguridad crítica, adicional a las métricas estándar de segmentación.

---

## División del trabajo

| Tarea | Responsable |
|---|---|
| Algoritmo A\* y generación del dataset | Angee Lorena Ocampo |
| Pipeline de preprocesamiento y carga | Angee Lorena Ocampo |
| Arquitectura U-Net y ablaciones | Daniel Usme Urrea |
| Arquitectura CNN | Daniela Escobar |
| Arquitectura GAN | Angee Lorena Ocampo |
| Visualización de resultados | Daniel Usme Urrea |
| Informe final | Todos |

---

## Fechas de entrega

| Entregable | Fecha límite |
|---|---|
| Propuesta | Miércoles 3 de junio de 2026, 11:59 p.m. |
| Check-in | Miércoles 10 de junio de 2026 |
| Deep Learning Day (presentación oral) | Miércoles 17 de diciembre de 2026 |
| Entrega final (informe + código + slides) | Miércoles 17 de diciembre de 2026, 11:59 p.m. |

---

## Referencias

> *(Completar con las referencias del estado del arte utilizadas en el informe)*

---

*Proyecto desarrollado en el marco del curso de Deep Learning — Ingeniería Electrónica, Universidad de Antioquia, 2026.*
