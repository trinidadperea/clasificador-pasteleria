import os
import json
import tensorflow as tf
import gradio as gr

# Rutas relativas a la carpeta donde está este archivo
BASE = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE, "modelo_pasteleria.keras"))
with open(os.path.join(BASE, "clases.json"), encoding="utf-8") as f:
    datos = json.load(f)

class_names = datos["class_names"]
traduccion = datos["traduccion"]
nombres_es = [traduccion[c] for c in class_names]


def predecir(img):
    if img is None:
        return None
    img = img[..., :3]                       # descarta el canal alfa si la imagen es PNG
    img = tf.image.resize(img, (224, 224))   # el preprocesamiento ya está dentro del modelo
    probs = model.predict(img[None, ...], verbose=0)[0]
    return {nombres_es[i]: float(probs[i]) for i in range(len(nombres_es))}


demo = gr.Interface(
    fn=predecir,
    inputs=gr.Image(label="Subí una foto"),
    outputs=gr.Label(num_top_classes=5, label="Predicción"),
    title="Clasificador de pastelería",
    description="Subí una foto de una tarta, torta, cupcake, dona o macaron.",
)

if __name__ == "__main__":
    demo.launch()
