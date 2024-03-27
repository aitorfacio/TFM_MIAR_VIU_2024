# TFM_MIAR_VIU_2024
Repositorio de código generado en la investigación de mi TFM para el Máster Universitario en Inteligencia Artifical de la VIU

# Proyecto de Tesis de Máster: Mitigación del Sesgo Racial en el Reconocimiento Facial

Este repositorio contiene una colección de scripts de Python desarrollados como parte de mi Tesis de Máster sobre la mitigación del sesgo racial en los sistemas de reconocimiento facial. El proyecto utiliza Redes Generativas Antagónicas (GANs) y técnicas de aprendizaje profundo para equilibrar los conjuntos de datos y mejorar la equidad de los modelos.

## Descripción General de los Scripts

### Clustering
- **`cluster_faces.py`**: Utiliza DBSCAN para agrupar codificaciones faciales, identificando caras únicas dentro del conjunto de datos.
- **`cluster_hierarchical.py`**: Aplica clustering jerárquico a codificaciones faciales para la categorización de identidades.
- **`cluster_kmeans.py`**: Implementa el clustering K-means en codificaciones faciales para organizar las caras en grupos.

### Preparación de Datos
- **`copy_n_images.py`**, **`copy_random_identities.py`**: Scripts para copiar números específicos de imágenes o identidades aleatorias entre directorios, ayudando en el equilibrio de los conjuntos de datos.
- **`create_benchmark_dataset.py`**, **`create_identity_set.py`**, **`create_imposter_genuine_lists.py`**, **`create_testing_dataset.py`**: Preparan conjuntos de datos para la evaluación del modelo, incluyendo la creación de subconjuntos y generación de pares genuinos-impostores.

### Visualización
- **`create_id_mosaic.py`**: Genera mosaicos a partir de imágenes de caras basadas en su identidad de agrupación.
- **`plot_arcface_loss_curves.py`**: Proporciona una visualización de las curvas de pérdida durante el entrenamiento de modelos basados en ArcFace. ArcFace es una técnica de reconocimiento facial que mejora la precisión del reconocimiento mediante la optimización de la distancia angular entre las embeddings, y este script ayuda a evaluar el desempeño y la convergencia del modelo.

- **`plot_cycle_gan_loss_curves.py`**: Este script genera gráficos de las curvas de pérdida para el entrenamiento de modelos CycleGAN. CycleGAN permite la transferencia de estilo entre dos dominios de imágenes sin necesidad de emparejamiento, siendo útil para la generación de imágenes sintéticas que ayudan a balancear los conjuntos de datos.

- **`plot_stylegan_loss_curves.py`**: Similar a los scripts anteriores de visualización, este se enfoca en graficar las curvas de pérdida para el entrenamiento de modelos StyleGAN. StyleGAN es conocido por su capacidad para generar imágenes de alta resolución y calidad, lo que lo hace ideal para la creación de conjuntos de datos sintéticos detallados.

### Utilidades
- **`docker_run.py`**: Facilita la ejecución de contenedores Docker con configuraciones específicas para entornos de investigación reproducibles.
- **`resize.py`**: Un script utilitario para redimensionar imágenes dentro de un directorio. Esto es especialmente útil para preparar conjuntos de datos, asegurando que todas las imágenes tengan las mismas dimensiones antes de ser procesadas por los modelos de aprendizaje profundo.
- **`generate.py`**: Este script está diseñado específicamente para la generación de imágenes sintéticas a través del uso de modelos CycleGAN preentrenados. Se centra en la transformación de imágenes de un dominio a otro, contribuyendo significativamente al objetivo de equilibrar los conjuntos de datos en términos de representación étnica. `generate.py` permite la creación de imágenes faciales sintéticas que mantienen las características esenciales de las identidades originales mientras se adaptan a las características de un nuevo grupo étnico. Esta capacidad lo hace invaluable para los esfuerzos de mitigación de sesgos raciales en los sistemas de reconocimiento facial, al permitir una representación más diversa y equitativa dentro de los conjuntos de datos utilizados para el entrenamiento de modelos de reconocimiento facial.

### Codificaciones y Embeddings
- **`embeddings.py`**, **`embeddings_deepface.py`**, **`encode_faces.py`**: Generan embeddings faciales utilizando diferentes modelos para un análisis facial más profundo.

### Análisis
- **`face_analysis.py`**, **`face_recognition_metrics.py`**: Realizan la detección de raza y calculan métricas de rendimiento como la Tasa de Error Igual (EER) para modelos de reconocimiento facial.

### Redes Generativas Antagónicas
- **`gan.py`**, **`gan_identity.py`**: Implementan y entrenan arquitecturas GAN para la generación de caras sintéticas para mejorar la diversidad del conjunto de datos. Extienden el fichero de https://github.com/eriklindernoren/PyTorch-GAN.
- **`my_dcgan.py`**: Implementa un modelo DCGAN (Deep Convolutional Generative Adversarial Network) personalizado. Este script es responsable de la creación, entrenamiento y generación de imágenes utilizando DCGAN, una variante de las GAN que utiliza convoluciones profundas, lo que lo hace especialmente adecuado para el procesamiento de imágenes.

### División de Conjuntos de Datos
- **`disjoint_datasets.py`**: Divide un conjunto de datos en dos subconjuntos disjuntos para fines de entrenamiento y validación.

## Objetivo

El objetivo de este proyecto es abordar el sesgo racial en las tecnologías de reconocimiento facial. Al mejorar la diversidad de los conjuntos de datos y las técnicas de agrupación, nos esforzamos por desarrollar sistemas de IA más equitativos y justos.

## Contribuciones

Este trabajo contribuye al campo de la IA ética proporcionando herramientas y métodos para evaluar y mejorar la equidad de los modelos de reconocimiento facial. Destaca la importancia del desarrollo responsable de la IA y la necesidad de mejoras continuas en la accesibilidad y la inclusión tecnológica.

