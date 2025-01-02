import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Vérifier si le fichier modèle existe avant de le charger
if not os.path.exists('dbscan_model.pkl'):
    st.error("Le fichier 'dbscan_model.pkl' est introuvable.")
else:
    # Charger le modèle DBSCAN
    with open('dbscan_model.pkl', 'rb') as file:
        dbscan = pickle.load(file)
    st.success("Modèle DBSCAN chargé avec succès.")

# Titre de l'application
st.title("Détection de Fraude")

# Charger vos données
uploaded_file = st.file_uploader("Téléchargez votre fichier de transactions (CSV)", type="csv")

if uploaded_file is not None:
    # Lire le fichier téléchargé
    data = pd.read_csv(uploaded_file)

    # Vérifiez que les colonnes nécessaires existent
    required_columns = ["TransactionAmount", "TransactionDate"]
    if all(col in data.columns for col in required_columns):
        # Vérifiez si les colonnes facultatives sont présentes avant de les afficher
        optional_columns = ["AccountID", "TransactionID", "LoginAttempts"]
        missing_optional_columns = [col for col in optional_columns if col not in data.columns]

        # Convertir TransactionDate en format datetime
        data["TransactionDate"] = pd.to_datetime(data["TransactionDate"])

        # Créer une nouvelle colonne "Durée" (différence en jours par rapport à la date minimale)
        data["Durée"] = (data["TransactionDate"] - data["TransactionDate"].min()).dt.days

        # Appliquer le modèle sur les colonnes pertinentes
        X = data[["TransactionAmount", "Durée"]]

        # Normaliser et standardiser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Utiliser DBSCAN pour prédire les clusters
        data["Cluster"] = dbscan.fit_predict(X_scaled)

        # Déterminer les transactions frauduleuses par DBSCAN
        data["Statut_DBSCAN"] = data["Cluster"].apply(lambda x: "Frauduleuse" if x == -1 else "Valide")

        # Combiner les statuts (si d'autres critères étaient présents, comme LoginAttempts)
        data["Statut"] = data["Statut_DBSCAN"]

        # Affichage des résultats
        st.subheader("Aperçu des résultats")
        display_columns = ["TransactionAmount", "TransactionDate", "Durée", "Cluster", "Statut"]
        for col in optional_columns:
            if col in data.columns:
                display_columns.insert(0, col)
        st.write(data[display_columns].head())

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
        st.error(f"Le fichier doit contenir les colonnes suivantes : {', '.join(required_columns)}.")
else:
    st.write("Veuillez télécharger un fichier CSV contenant vos données de transactions.")
