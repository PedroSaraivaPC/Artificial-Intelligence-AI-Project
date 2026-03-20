import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Classificador de Carroçarias",
    page_icon="🚗"
)

# css do titulo principal e do card de background do: "🔍 Classe Predita:"
st.markdown("""
<style>
    .title-text {
        font-size: 40px !important;
        font-weight: 700 !important;
        color: black;
    }
    .pred-card {
        background: #f0f6ff;
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 6px solid #1a4fff;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# carrega o modelo .keras
model = tf.keras.models.load_model(
    r"C:/Users/Pedro/Desktop/ISEC/3o_Ano/IC/TrabalhoPratico/MetasTrabalho/Meta3/final_model.keras"
)
class_names = ["Convertible", "Coupe", "SUV", "Sedan", "Van"]

# titulo
st.markdown('<p class="title-text">Classificador de Carroçarias – Meta III</p>', unsafe_allow_html=True)

st.image(r"C:/Users/Pedro/Desktop/ISEC/3o_Ano/IC/TrabalhoPratico/MetasTrabalho/Meta3/imagem_inicio.jpg")

# escolha do metodo de input
input_mode = st.radio(
    "📸 Escolha o método de entrada para descobrir qual o tipo de carroçaria:",
    ["Upload de imagem", "Usar câmara"],
    horizontal=True
)

uploaded_file = None
camera_image = None

# caso seja upload de imagem
if input_mode == "Upload de imagem":
    uploaded_file = st.file_uploader(
        "📁 Upload da imagem",
        type=["jpg", "jpeg", "png"]
    )
# caso seja usar camara
elif input_mode == "Usar câmara":
    camera_image = st.camera_input("🤳 Tire uma fotografia")

# processamento da imagem
if uploaded_file or camera_image:

    if uploaded_file:
        img = Image.open(uploaded_file)

    elif camera_image:
        img = Image.open(camera_image)

    # Mostrar imagem carregada
    img = img.convert("RGB").resize((224,224))
    st.image(img, caption="Imagem carregada", width=350)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # predict da imagem
    pred = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(pred)]
    pred_percent = pred[np.argmax(pred)] * 100

    # card do: "🔍 Classe Predita:"
    st.markdown(
        f"""
        <div class="pred-card">
            <h3>🔍 Classe Predita:</h3>
            <h2><b>{pred_class}</b> ({pred_percent:.2f}% de confiança)</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    st.subheader("📊 Probabilidades por classe:")

    df = pd.DataFrame({
        "Classe": class_names,
        "Probabilidade": pred
    })

    # grafico de barras de probabilidades
    st.bar_chart(df.set_index("Classe"))

    # valores numericos das probabilidades em tabela
    st.write("Valores numéricos das probabilidades:")
    st.dataframe(df)
