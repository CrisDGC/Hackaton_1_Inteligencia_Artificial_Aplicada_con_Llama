# Tutor de Papers Científicos — RAG + Fine-Tuning (LoRA)

Un asistente que ayuda a estudiar artículos científicos sección por sección: clasifica el rol retórico de cada fragmento, busca el dato exacto en el paper o en un glosario curado, y redacta una explicación pedagógica citando siempre de dónde sale la información.

**Está especializado en temas de Machine Learning e Inteligencia Artificial** — tanto el glosario de conceptos como el dataset del clasificador están construidos sobre terminología de ML/IA, así que el tutor rinde mejor con papers de esa área.

## Qué combina el proyecto

| Pieza | Técnica | Qué hace |
|---|---|---|
| RAG interno | `sentence-transformers` | Indexa el PDF subido por secciones/oraciones para recuperar el fragmento relevante a cada pregunta |
| RAG externo | `sentence-transformers` | Busca definiciones en un glosario curado de ML/IA, con umbral de similitud para admitir honestamente cuando un término no está cubierto |
| Fine-tuning clasificador | DistilBERT multilingüe + LoRA | Etiqueta cada fragmento con su rol retórico (OBJETIVO, METODOLOGIA, RESULTADO, DISCUSION, etc.) |
| Fine-tuning generativo (demo) | TinyLlama + LoRA (`SFTTrainer`) | Ajusta el tono de un modelo generativo pequeño, como demostración independiente de LoRA sobre un modelo causal |
| Generación | Groq (`openai/gpt-oss-20b`) | Redacta la explicación final, citando si el dato viene del paper, del glosario o de conocimiento general |
| Evaluación | Matriz de confusión + Groq-juez | Mide qué tan bien clasifica el rol retórico y si el tutor es honesto sobre sus fuentes |

## Requisitos

- Cuenta gratuita en [groq.com](https://groq.com) para obtener una `GROQ_API_KEY`.
- Google Colab (el notebook está pensado para correr ahí).
- GPU solo es necesaria para el **Paso 9.5** (demo de LoRA generativo sobre TinyLlama) — el resto del notebook corre en CPU.

## Cómo correrlo en Google Colab

### 1. Configura tu API key de Groq en Colab Secrets

1. Abre el notebook en Colab.
2. Haz clic en el ícono de llave 🔑 en la barra lateral izquierda.
3. Agrega un secreto nuevo con el nombre exacto `GROQ_API_KEY` y pega tu llave (la obtienes gratis en groq.com → API Keys).
4. Activa el acceso a ese secreto para este notebook (el switch junto al nombre del secreto).

La llave nunca se escribe directamente en el código — el notebook la lee desde Secrets.

### 2. Corre las celdas en orden, de arriba hacia abajo

Cada celda depende de las anteriores (embeddings, modelo, dataset, etc.), así que no saltes pasos la primera vez.

### 3. Prueba con el PDF de ejemplo incluido en este repositorio

Este repo incluye un PDF de prueba (`An_Overview_of_Neural_Network.pdf`)  para que puedas correr el notebook de principio a fin sin buscar tu propio artículo:

1. Descarga el PDF desde este repositorio.
2. Súbelo a la sesión de Colab arrastrándolo al panel de archivos (ícono de carpeta 📁 en la barra lateral izquierda), **o** usa el botón "Sube tu PDF 📎" de la app de Gradio al final del notebook.

### 4. Si quieres experimentar con otro paper

Tienes dos formas de hacerlo:

- **Desde la app de Gradio (recomendado, sin tocar código):** al llegar al último paso del notebook, la interfaz tiene un botón para subir cualquier PDF directamente — el sistema lo indexa automáticamente.
- **Desde el notebook, de forma manual (Paso 3):** sube tu PDF a Colab y cambia la variable `ruta` por el nombre de tu archivo:

  ```python
  ruta = "/content/tu_paper.pdf"
  ```

## Notas sobre el alcance

- El clasificador de rol retórico y el modelo de tono con LoRA se entrenaron con datasets pequeños (30-50 ejemplos) — son demostraciones de la técnica, no modelos listos para producción.
- El modelo de tono afinado con LoRA (Paso 9.5) **no se usa** en el pipeline final — el sistema real redacta siempre con Groq; ese paso solo demuestra el fine-tuning generativo de forma aislada.
- La matriz de confusión y las métricas del juez se calculan sobre un set de prueba fijo y chico — son un ejemplo, no una medición estadísticamente robusta.
