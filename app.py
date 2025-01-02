import pickle
import streamlit as st
import pandas as pd

# Charger le modèle DBSCAN
with open('dbscan_model.pkl', 'rb') as file:
    dbscan = pickle.load(file)

# Titre de l'application
st.title("Détection de Fraude")

# Charger vos données
uploaded_file = st.file_uploader("Téléchargez votre fichier de transactions (CSV)", type="csv")

if uploaded_file is not None:
    # Lire le fichier téléchargé
    data = pd.read_csv(uploaded_file)

    # Vérifiez que les colonnes nécessaires existent
    if all(col in data.columns for col in ["TransactionAmount", "TransactionDate", "LoginAttempts"]):
        # Convertir TransactionDate en format datetime
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])

        # Créer une nouvelle colonne "Durée" (différence en jours par rapport à la date minimale)
        data["Durée"] = (data["TransactionDate"] - data["TransactionDate"].min()).dt.days

        # Appliquer le modèle sur les colonnes pertinentes
        X = data[["TransactionAmount", "Durée"]]
        data["Cluster"] = dbscan.fit_predict(X)

        # Déterminer les transactions frauduleuses par DBSCAN
        data["Statut_DBSCAN"] = data["Cluster"].apply(lambda x: "Frauduleuse" if x == -1 else "Valide")

        # Vérifier LoginAttempts
        mean_login_attempts = 1.125152
        data["Statut_Login"] = data["LoginAttempts"].apply(
            lambda x: "Frauduleuse" if x > mean_login_attempts else "Valide"
        )

        # Combiner les statuts
        data["Statut"] = data.apply(
            lambda row: "Frauduleuse" if row["Statut_DBSCAN"] == "Frauduleuse" or row["Statut_Login"] == "Frauduleuse"
            else "Valide",
            axis=1,
        )

        # Affichage des résultats
        st.subheader("Aperçu des résultats")
        st.write(data[["TransactionAmount", "TransactionDate", "Durée", "LoginAttempts", "Cluster", "Statut"]].head())

        # Téléchargement des résultats
        st.subheader("Télécharger les résultats")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les résultats au format CSV",
            data=csv,
            file_name="transactions_analyzed.csv",
            mime="text/csv"
        )
    else:
        st.error("Le fichier doit contenir les colonnes 'TransactionAmount', 'TransactionDate' et 'LoginAttempts'.")
else:
    st.write("Veuillez télécharger un fichier CSV contenant vos données de transactions.")
