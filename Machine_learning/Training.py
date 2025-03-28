import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("data", exist_ok=True)


#  Chemin du fichier Excel corrigé
file_path = r"C:/Users/hp/Documents/GitHub/pediatric-appendicitis-diagnosis/data/cleaned_data.xlsx"

#  Vérifier que le fichier existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Fichier non trouvé : {file_path}")

#  Charger le fichier Excel
dataset = pd.read_excel(file_path)

print(" Dataset chargé avec succès !")
print(dataset.head())
print("Colonnes disponibles :", dataset.columns)

#  Vérification de la colonne cible
target_col = "Diagnosis"
if target_col not in dataset.columns:
    raise ValueError(f"❌ La colonne cible '{target_col}' est introuvable dans le dataset.")

#  Définir X (features) et y (target)
X = dataset.drop(columns=[target_col])
y = dataset[target_col]

#  Encoder la variable cible si elle est catégorique
if y.dtype == 'object' or len(y.unique()) > 2:
    y = LabelEncoder().fit_transform(y)

#  Encoder les colonnes catégoriques
categorical_cols = X.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

#  Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Train: {X_train.shape}, Test: {X_test.shape}")

#  Sauvegarder les données AVANT normalisation (pour SHAP)
X_train_original = X_train.copy()
X_test_original = X_test.copy()

#  Appliquer la normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_original.to_csv("data/X_train_original.csv", index=False)
X_test_original.to_csv("data/X_test_original.csv", index=False)

#  Optimisation de LightGBM avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 10],
    'num_leaves': [20, 31, 40],
    'min_child_samples': [5, 10, 20]
}

print("➡ Optimisation des hyperparamètres de LightGBM...")
lgbm = LGBMClassifier(random_state=42)

#  Lancer la recherche GridSearchCV
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

print(" Meilleurs paramètres LightGBM :", grid_search.best_params_)
print(" Modèle LightGBM optimisé entraîné avec succès !")

#  Définir les autres modèles
models = {
    "SVM": SVC(probability=True, kernel="linear", random_state=42),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "LightGBM Optimisé": best_model
}

#  Entraîner et évaluer chaque modèle
results = {}

for name, model in models.items():
    print(f"\n➡ Entraînement du modèle {name}...")
    try:
        model.fit(X_train, y_train)
        print(f" {name} entraîné avec succès !")

        #  Faire des prédictions
        y_pred = model.predict(X_test)

        #  Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        #  Vérifier si predict_proba fonctionne pour calculer l'AUC-ROC
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        #  Ajouter les résultats dans le dictionnaire
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "AUC-ROC": auc
        }

    except Exception as e:
        print(f" Erreur avec {name} : {e}")

#  Afficher les résultats finaux
print("\n Résultats de l'évaluation des modèles :")
for model, metrics in results.items():
    print(f"\n📌 {model}:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")

#  Sauvegarder les fichiers après normalisation
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

X_train_df.to_csv("data/X_train.csv", index=False)
X_test_df.to_csv("data/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)

#  Sauvegarder le modèle et le scaler
joblib.dump(best_model, "data/best_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print(" Données d'entraînement et de test sauvegardées avec succès !")
