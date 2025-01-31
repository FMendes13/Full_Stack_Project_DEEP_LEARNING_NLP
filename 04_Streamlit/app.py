import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration de la page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🤫",
    layout="wide"  # Utiliser une mise en page large
)

# URL des données et des GIFs
DATA_URL = 'https://fnd-jedha-project.s3.eu-west-3.amazonaws.com/WelFake_data_app.csv'
FAKE_NEWS_GIF = 'https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjgyMDIwazU0ZmsxOXozYnN4NTd0Nml6aDk1N3cxbmZjd3hvazZxcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iyau7QcKtKUYIda/giphy.gif'
REAL_NEWS_GIF = 'https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZm4ybnFsOHJ3MTdyNWM2b25kcm1lem5oN3h0MXVlNTV6ajl1ZzZ0MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ap6wcjRyi8HoA/giphy.gif'

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

# Entraînement du modèle
@st.cache_data
def train_model(data):
    X = data['clean_token']
    y = data['label']

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
    X = text_transformer.fit_transform(X)

    lr = LogisticRegression(solver='lbfgs', random_state=17)
    lr.fit(X, y)

    return lr, text_transformer

# Fonction pour prédire si une news est fake ou réelle
def predict_news(text, model, transformer, threshold=0.5):
    input_transformed = transformer.transform([text])
    prediction_proba = model.predict_proba(input_transformed)
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction, prediction_proba

# Initialisation des états de session
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'show_probability' not in st.session_state:
    st.session_state.show_probability = False

# Chargement des données et entraînement du modèle
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Ready to Go")

lr, text_transformer = train_model(data)

# Appliquer un thème sombre
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;  /* Fond noir */
        color: #FFFFFF;            /* Texte blanc */
    }
    .stTextArea textarea {
        background-color: #1E1E1E; /* Fond sombre pour la zone de texte */
        color: #FFFFFF;            /* Texte blanc */
        width: 100%;               /* Largeur maximale */
        max-width: 1200px;         /* Largeur maximale limitée */
        margin: auto;              /* Centrer horizontalement */
        height: 200px;             /* Hauteur de la zone de texte */
    }
    .stButton button {
        background-color: #4CAF50; /* Bouton vert */
        color: #FFFFFF;            /* Texte blanc */
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049; /* Vert plus foncé au survol */
    }
    .css-1d391kg {
        color: #FFFFFF;            /* Texte blanc pour les titres */
    }
    .fake-news {
        color: red;
    }
    .real-news {
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal
st.markdown(
    """
    <h1 style="text-align:center; color:#FFFFFF; font-size:40px;">
        🤖 Fake News Detector 🔍
    </h1>
    """,
    unsafe_allow_html=True
)

# Texte d'introduction (clair et concis)
st.markdown("""
    <div style="text-align:center; font-size:18px; color:#FFFFFF; padding:20px;">
        <p style="font-weight:bold; color:#4CAF50;">
            🚨 In a world full of misinformation, can you trust what you read? 🚨
        </p>
        <p>
            This tool uses advanced machine learning to help you detect whether a news article is <strong>real</strong> or <strong>fake</strong>.
        </p>
        <p>
            Simply paste the text below and let the model do the work!
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

# Séparateur
st.markdown("---")

# Conteneur pour centrer les éléments
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])  # Colonnes pour centrer le contenu
    with col2:
        # Zone de texte pour entrer ce que l'utilisateur veut vérifier
        input_client = st.text_area(
            "Enter the news text you want to check:",
            placeholder="Paste the news article here...",
            height=200  # Hauteur de la zone de texte
        )

        # Bouton d'analyse
        if st.button('🔍 Analyze', use_container_width=True):
            if input_client:
                with st.spinner("Analyzing..."):
                    st.session_state.input_text = input_client
                    st.session_state.prediction, st.session_state.prediction_proba = predict_news(
                        input_client, lr, text_transformer, threshold=0.3
                    )
            else:
                st.error("⚠️ Please enter some text to analyze.")

        # Bouton pour afficher les probabilités
        if st.button('Show Probability', use_container_width=True):
            st.session_state.show_probability = True

# Affichage des résultats
if st.session_state.prediction is not None:
    st.markdown("---")
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            if st.session_state.prediction.size > 0:
                if (st.session_state.prediction == 0).any():
                    st.markdown('<p class="fake-news">🚨 **Fake News Detected** 🚨</p>', unsafe_allow_html=True)
                    st.image(FAKE_NEWS_GIF, use_container_width=True)
                else:
                    st.markdown('<p class="real-news">✔️ **This seems legit** ✔️ - yet do not hesitate to verify the sources</p>', unsafe_allow_html=True)
                    st.image(REAL_NEWS_GIF, use_container_width=True)

            # Affichage des probabilités si l'utilisateur le souhaite
            if st.session_state.show_probability:
                max_prob = max(st.session_state.prediction_proba[0])
                if max_prob == st.session_state.prediction_proba[0][0]:
                    st.subheader(f'Fake News Probability: {max_prob * 100:.2f}%')
                else:
                    st.subheader(f'Real News Probability: {max_prob * 100:.2f}%')

# Footer avec le lien GitHub
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#FFFFFF;">
        📢 *NLP Project - Fake News Detection - 2025*<br>
        Created by : Fredéric, Yannick et Mohamed <br>
        Find the code on <a href="https://github.com/FMendes13/Full_Stack_Project_DEEP_LEARNING_NLP" target="_blank" style="color:#4CAF50;">GitHub</a>.
    </div>
    """,
    unsafe_allow_html=True
)
