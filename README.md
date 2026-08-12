# 🎬 Sistema Inteligente para la Clasificación de Críticas de Películas (NLP & Sentiment Analysis)

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Keras](https://img.shields.io/badge/Keras-Preprocessing-red?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

> **Un sistema inteligente basado en Procesamiento del Lenguaje Natural (NLP) y Aprendizaje Automático capaz de analizar y clasificar reseñas de películas en español como POSITIVAS o NEGATIVAS, proporcionando retroalimentación estratégica para la industria cinematográfica y cadenas de cine.**

---

## 📋 Tabla de Contenidos
- [📌 Visión General y Justificación](#-visión-general-y-justificación)
- [🚨 Situación Problemática](#-situación-problemática)
- [🎯 Objetivos del Proyecto](#-objetivos-del-proyecto)
- [💡 En Qué Ayuda este Proyecto (Impacto de Negocio)](#-en-qué-ayuda-este-proyecto-impacto-de-negocio)
- [🛠️ Arquitectura y Metodología KDD](#️-arquitectura-y-metodología-kdd)
- [🔬 Preprocesamiento de Texto y NLP](#-preprocesamiento-de-texto-y-nlp)
- [⚙️ Modelado y Codificación](#️-modelado-y-codificación)
- [📊 Resultados y Evaluación](#-resultados-y-evaluación)
- [🔗 Recursos y Enlaces del Proyecto](#-recursos-y-enlaces-del-proyecto)
- [👤 Autor](#-autor)

---

## 📌 Visión General y Justificación

El **Análisis de Sentimientos (*Sentiment Analysis*)** es una subdisciplina del Procesamiento del Lenguaje Natural (NLP) que permite identificar automáticamente la polaridad emocional detrás de un texto. Esta técnica permite clasificar si una opinión, reseña o comentario expresa una actitud **positiva**, **negativa** o **neutral** hacia un tema específico.

### 🎯 Justificación del Proyecto

Inspirado en estudios clásicos de clasificación sobre conjuntos de datos de IMDb, este proyecto adapta y optimiza algoritmos de Machine Learning para procesar un corpus de **8,603 críticas de usuarios en castellano** provenientes del sitio web **FilmAffinity**.

### 🚀 Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│   │   Reseña en     │    │  Preprocesamiento│    │  Vectorización  │                │
│   │   Texto Plano   │───▶│       NLP       │───▶│  TF-IDF/Binary  │                │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘                │
│            │                      │                       │                        │
│            ▼                      ▼                       ▼                        │
│   ┌─────────────────────────────────────────────────────────────────┐               │
│   │                                                                 │               │
│   │              Clasificador Binario (ML)                          │               │
│   │                                                                 │               │
│   └─────────────────────────────────────────────────────────────────┘               │
│                              │                                                      │
│                              ▼                                                      │
│   ┌─────────────────────────────────────────────────────────────────┐               │
│   │                                                                 │               │
│   │              1: Positivo  /  0: Negativo                        │               │
│   │                                                                 │               │
│   └─────────────────────────────────────────────────────────────────┘               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 Situación Problemática

* **Incertidumbre en la Elección:** Las salas de cine y las plataformas de streaming enfrentan el reto de recomendar contenido adecuado. Los espectadores se guían por reseñas masivas en redes sociales y foros, pero la sobreinformación dificulta identificar la recepción real de una película.
* **Falta de Feedback Estructurado:** Las productoras audiovisuales reciben miles de comentarios desestructurados, perdiendo la oportunidad de extraer métricas cuantitativas sobre qué elementos (guión, actuaciones, dirección) agradaron o desagradaron al público.

---

## 🎯 Objetivos del Proyecto

* **Objetivo Principal:** Desarrollar un sistema inteligente para reconocer y clasificar automáticamente críticas y reseñas de películas en lenguaje castellano en categorías binarias (Positivo / Negativo).
* **Objetivos Específicos:**
  1. Construir un pipeline de preprocesamiento de texto (*Stopwords*, *Stemming*, *Tokenización*).
  2. Implementar esquemas de vectorización y representación de texto (*Bag of Words*, *TF-IDF*, *Binary Encoding*).
  3. Entrenar y evaluar modelos supervisados de clasificación (Regresión Logística / Naive Bayes).

---

## 💡 En Qué Ayuda este Proyecto (Impacto de Negocio)

1. **Optimización para Cines y Cadenas de Distribución:**
   * Permite ajustar la cartelera y las estrategias de promoción en tiempo real según la respuesta emocional del público.
2. **Feedback Automatizado para Productoras Cinematográficas:**
   * Proporciona un panel analítico a los directores y productores sobre la recepción masiva de sus obras para mejorar futuras producciones.
3. **Sistemas de Recomendación Personalizada:**
   * Sirve como motor base para filtrar y recomendar películas a los usuarios en función de las críticas con sentimientos positivos.

---

## 🛠️ Arquitectura y Metodología KDD

El proyecto se desarrolló siguiendo la metodología **KDD (*Knowledge Discovery in Databases*)**, un proceso iterativo e interactivo para la extracción de conocimiento a partir de grandes volúmenes de datos.

### 📊 Diagrama de Flujo KDD

```
+------------------+     +--------------------+     +-------------------+     +------------------+     +-------------------+
|  1. Selección    | --> | 2. Preprocesamiento| --> | 3. Transformación | --> | 4. Minería de    | --> | 5. Evaluación e   |
|     de Datos     |     |     y Limpieza     |     |    y Vectorización|     |    Datos (ML)    |     |    Interpretación |
+------------------+     +--------------------+     +-------------------+     +------------------+     +-------------------+
         |                        |                         |                     |                     |
         ▼                        ▼                         ▼                     ▼                     ▼
+------------------+     +--------------------+     +-------------------+     +------------------+     +-------------------+
| • Carga de CSV   |     | • Eliminación de   |     | • Tokenización    |     | • Entrenamiento   |     | • Matriz de       |
| • Exploración    |     |   caracteres       |     | • Stemming        |     |   de Modelos      |     |   Confusión       |
|   inicial        |     |   especiales       |     | • Stopwords       |     | • GridSearch CV   |     | • Accuracy        |
| • Identificación |     | • Normalización    |     | • TF-IDF          |     | • Regresión       |     | • Interpretación  |
|   de columnas    |     |   de texto         |     |   Vectorización   |     |   Logística       |     |   de Features     |
+------------------+     +--------------------+     +-------------------+     +------------------+     +-------------------+
```

1. **Selección de Datos:** Extracción de 8,603 críticas de FilmAffinity.
2. **Preprocesamiento:** Eliminación de caracteres especiales, conversión a minúsculas, remoción de *stopwords* en español e impartición de *Stemming* (SnowballStemmer).
3. **Transformación:** Representación matricial mediante `TfidfVectorizer` y matriz binaria con Keras.
4. **Minería de Datos:** Entrenamiento supervisado dividiendo el dataset en 67% Train / 33% Test estratificado.
5. **Evaluación:** Medición mediante Matriz de Confusión y prueba de predicción en tiempo real.

---

## 🔬 Preprocesamiento de Texto y NLP

La limpieza profunda garantizó que solo las palabras con carga semántica aportaran al modelo:

* **Tokenización:** Uso de `ToktokTokenizer` para segmentar palabras.
* **Stopwords Removal:** Eliminación de palabras vacías en español (*de, la, que, el, en, a, etc.*) mediante NLTK.
* **Stemming:** Reducción de palabras a su raíz canónica usando `SnowballStemmer("spanish")` (*ejemplo: "divertidísima", "divertido" ➡️ "divert"*).

---

## ⚙️ Modelado y Codificación

| Parámetro / Componente | Configuración |
| :--- | :--- |
| **Dataset Total** | 8,603 críticas (3,920 Positivas / 4,683 Negativas) |
| **Split de Datos** | 67% Entrenamiento (5,764) / 33% Prueba (2,839) |
| **Vectorización** | TF-IDF (`max_features=20,000`) & Matrix Mode Binary |
| **Algoritmo Base** | Regresión Logística / Naive Bayes (Teorema de Bayes) |
| **Etiquetado** | `1`: Rating > 6 (Positivo) \| `0`: Rating ≤ 5 (Negativo) |

---

## 📊 Resultados y Evaluación

El modelo es capaz de predecir la polaridad de oraciones complejas e inéditas con alta precisión:

```python
# Ejemplo de prueba en tiempo real:
opinion = ["La película me desagrada"]
X_prueba = tfidf.transform(opinion)
best_clf.predict(X_prueba)

# Output: array([0]) -> Clasificado como NEGATIVO
```

---

## 🔗 Recursos y Enlaces del Proyecto

* 📓 **Notebook ejecutable en Google Colab:** [Ver Colab Notebook](https://colab.research.google.com/drive/1SngWKXuVj2WZjuavWiYEUIkd4pFeo025?usp=sharing)
* 📊 **Presentación del Proyecto en Google Slides:** [Ver Diapositivas PPTX](https://docs.google.com/presentation/d/1ZPTA1cxhn6ut32QMvqB7DP2ej_Izg5MuyAykTkewTbU/edit?usp=sharing)
* 🗃️ **Dataset Original en Kaggle:** [Críticas Películas FilmAffinity en Español](https://www.kaggle.com/datasets/ricardomoya/criticas-peliculas-filmaffinity-en-espaniol) *(Guardar como `proyecto.csv`)*

---

## 👤 Autor

Desarrollado por **Maykol Anthony Vargas Bringas**  
*Systems Engineer & Data Analyst*
