import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Détecteur de Fraude", layout="centered")
st.title("💳 Détecteur de Fraude par Carte de Crédit")

# Charger le modèle sauvegardé
model = joblib.load("models/random_forest.joblib")

uploaded_file = st.file_uploader("Dépose ici un fichier CSV contenant des transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données :")
    st.write(df.head())

    # S'assurer que 'Class' n'est pas dans les données qu'on prédit
    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # Prédictions
    prediction = model.predict(df)
    df['Prediction'] = prediction

    st.subheader("Résultats de la prédiction :")
    st.dataframe(df[['Prediction']])
