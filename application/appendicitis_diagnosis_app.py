import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib

#  Charger le modèle LightGBM et le scaler
model = joblib.load("data/best_model.pkl")
scaler = joblib.load("data/scaler.pkl")

#  Charger les colonnes originales utilisées lors de l'entraînement
original_features = pd.read_csv("data/X_train_original.csv").columns

#  Interface utilisateur
st.title("🩺 Prédiction de l'Appendicite chez l'Enfant")

#  Dictionnaires de conversion
sex_mapping = {"Masculin": 0, "Féminin": 1}
boolean_mapping = {"Non": 0, "Oui": 1}

#  Entrées utilisateur
inputs = {
    "Age": st.number_input("Âge de l'enfant", min_value=0, max_value=18, value=10),
    "Sex": sex_mapping[st.selectbox("Sexe", ["Masculin", "Féminin"])],
    "Migratory_Pain": boolean_mapping[st.selectbox("Douleur migratoire", ["Non", "Oui"])],
    "Lower_Right_Abd_Pain": boolean_mapping[st.selectbox("Douleur au bas droit de l'abdomen", ["Non", "Oui"])],
    "Coughing_Pain": boolean_mapping[st.selectbox("Douleur en toussant", ["Non", "Oui"])],
    "Nausea": boolean_mapping[st.selectbox("Nausée", ["Non", "Oui"])],
    "Loss_of_Appetite": boolean_mapping[st.selectbox("Perte d'appétit", ["Non", "Oui"])],
    "Body_Temperature": st.number_input("Température corporelle (°C)", min_value=35.0, max_value=42.0, value=37.0),
    "WBC_Count": st.number_input("Nombre de globules blancs (10^9/L)", min_value=0.0, value=10.0),
    "Neutrophil_Percentage": st.number_input("Pourcentage de neutrophiles", min_value=0.0, max_value=100.0, value=50.0),
    "Neutrophilia": boolean_mapping[st.selectbox("Neutrophilie", ["Non", "Oui"])],
    "RBC_in_Urine": boolean_mapping[st.selectbox("Présence de globules rouges dans les urines", ["Non", "Oui"])],
    "WBC_in_Urine": boolean_mapping[st.selectbox("Présence de globules blancs dans les urines", ["Non", "Oui"])],
    "CRP": st.number_input("CRP (mg/L)", min_value=0.0, value=5.0),
    "Dysuria": boolean_mapping[st.selectbox("Dysurie", ["Non", "Oui"])],
    "Stool": boolean_mapping[st.selectbox("Selles anormales", ["Non", "Oui"])],
    "Peritonitis": boolean_mapping[st.selectbox("Signes de péritonite", ["Non", "Oui"])],
    "Psoas_Sign": boolean_mapping[st.selectbox("Signe du psoas", ["Non", "Oui"])],
    "Ipsilateral_Rebound_Tenderness": boolean_mapping[st.selectbox("Signe de rebond ipsilatéral", ["Non", "Oui"])]
}

#  Prédiction avec le modèle
if st.button("Prédire"):
    input_df = pd.DataFrame([inputs])

    #  Ajouter automatiquement les colonnes manquantes et trier dans le bon ordre
    for col in original_features:
        if col not in input_df.columns:
            input_df[col] = 0

    #  Supprimer les colonnes inutilisées
    input_df = input_df[original_features]

    #  Appliquer la normalisation
    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    #  Prédiction (en probabilité)
    proba = model.predict(input_df)[0]
    prediction = int(proba > 0.5)

    #  Affichage du résultat
    st.markdown(f"### 🏥 **Prédiction :** {' Appendicite' if prediction == 1 else '❌ Non Appendicite'}")
    st.markdown(f"### 🔎 **Probabilité d'appendicite :** {proba:.2%}")

    #  Interprétation SHAP
    try:
        st.subheader("📊 Facteurs influençant la décision (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        #  Affichage des valeurs SHAP sous forme de graphique waterfall
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             feature_names=input_df.columns))
        st.pyplot(fig)

        #  Affichage sous forme de graphique force plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            matplotlib=True,
            show=False
        )
        plt.savefig("force_plot.png", bbox_inches='tight')
        st.image("force_plot.png")

        #  Affichage sous forme de graphique bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.bar_plot(shap_values[0], feature_names=input_df.columns, max_display=10)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de l'interprétation SHAP : {e}")

