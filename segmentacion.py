import streamlit as st
import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

st.title("Segmentación de Imágenes by Lucas De Rito")

# Subir imagen
uploaded_file = st.file_uploader("Sube imagen para segmentar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Leer la imagen
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Aplicar un filtro de desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar la transformación de Sobel para detectar bordes
    edges = cv2.Canny(blurred, 50, 150)

    # Operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distancia euclidiana y detección de marcadores
    distance = ndimage.distance_transform_edt(closing)
    # Detectar picos locales y convertirlos en una máscara del mismo tamaño
    local_max = peak_local_max(distance, min_distance=20, labels=closing, num_peaks=np.inf)
    markers = np.zeros_like(distance, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, local_max.shape[0] + 1)  # Etiquetas únicas para cada marcador

    labels = watershed(-distance, markers, mask=closing)

    # Dibujar contornos sobre la imagen original
    segmented_image = image.copy()
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(segmented_image, contours, -1, (255, 0, 0), 2)

    # Mostrar resultados en Streamlit
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Imagen Original", use_container_width=True)
    with col2:
        st.image(edges, caption="Detección de Bordes", use_container_width=True, channels="GRAY")
    with col3:
        st.image(segmented_image, caption="Segmentación con Watershed", use_container_width=True)


st.markdown('---')
st.markdown("""
## Descripción de la Aplicación

Esta aplicación utiliza técnicas de **detección de bordes** y **segmentación con Watershed** para analizar y dividir una imagen en regiones homogéneas.

### Técnicas utilizadas
- **Detección de Bordes (Canny):**  
  Resalta los contornos principales de los objetos en la imagen.
- **Segmentación con Watershed:**  
  Emplea la transformada de distancia para separar y etiquetar las distintas regiones de la imagen.

### Casos de uso

#### Diagnóstico Médico y Análisis de Imágenes Médicas
- **Segmentación de órganos y tejidos:**  
  Permite aislar y analizar estructuras específicas en imágenes de resonancia magnética (MRI) o tomografías computarizadas (CT).
- **Detección de tumores o anomalías:**  
  Ayuda a identificar y delimitar regiones sospechosas, facilitando el diagnóstico y la planificación del tratamiento.

#### Inspección de Calidad Industrial
- **Detección de defectos en productos:**  
  Utiliza la segmentación para identificar imperfecciones o irregularidades en la producción de componentes electrónicos, piezas mecánicas o materiales.
- **Control de procesos:**  
  Segmenta imágenes de productos en línea para asegurar que cumplan con los estándares de calidad.

#### Automoción y Vehículos Autónomos
- **Segmentación de la escena vial:**  
  Distingue entre carreteras, peatones, vehículos y señales de tráfico.
- **Detección de obstáculos:**  
  La detección de bordes ayuda a identificar límites y contornos esenciales para la navegación segura.

#### Agricultura de Precisión
- **Monitoreo de cultivos:**  
  Segmenta imágenes aéreas o de drones para distinguir entre cultivos, malezas y suelo, permitiendo un análisis detallado del crecimiento y la salud de las plantas.
- **Detección de plagas o enfermedades:**  
  Aísla áreas afectadas en imágenes de campos agrícolas.

#### Análisis de Imágenes Satelitales y Aéreas
- **Mapeo y clasificación del uso del suelo:**  
  Segmenta diferentes tipos de terrenos, cuerpos de agua, áreas urbanas y vegetación.
- **Monitoreo ambiental:**  
  Detecta cambios en la cobertura forestal o en cuerpos de agua a lo largo del tiempo.

#### Procesamiento y Mejoramiento de Imágenes
- **Efectos artísticos y edición fotográfica:**  
  Permite aplicar efectos o filtros de forma selectiva sobre partes de una imagen (por ejemplo, desenfocar el fondo mientras se mantiene el sujeto enfocado).
- **Reconstrucción y restauración de imágenes:**  
  Al aislar ciertas regiones, se pueden aplicar técnicas de restauración de manera más precisa.

#### Robótica y Automatización
- **Reconocimiento y manipulación de objetos:**  
  Los robots pueden segmentar el entorno para identificar y agarrar objetos con mayor precisión, facilitando tareas de ensamblaje o clasificación.
- **Navegación en entornos complejos:**  
  Utiliza la segmentación para identificar caminos y obstáculos en entornos no estructurados.

Esta aplicación es parte del portafolio de proyectos de Lucas De Rito, demostrando habilidades en inteligencia artificial, visión por computadora y procesamiento de imágenes.

*Desarrollada con Streamlit, OpenCV y Scikit-Image.*
""")
