Diagnostic de l’Appendicite Pédiatrique


   1. Vue d’ensemble
Ce projet vise à diagnostiquer l’appendicite pédiatrique à l’aide de modèles de machine learning. Le flux de travail comprend :

✔ Nettoyage et prétraitement des données (gestion des valeurs manquantes, des valeurs aberrantes et des corrélations).
✔ Sélection des caractéristiques pour conserver les attributs médicaux les plus pertinents.
✔ Entraînement et évaluation des modèles de ML pour déterminer le plus performant.
✔ Déploiement d’une application web permettant aux professionnels de santé d’entrer des données patients et d’obtenir des prédictions.
✔ Interprétation des résultats avec SHAP pour expliquer les décisions du modèle.
✔ Optimisation de la mémoire pour améliorer les performances.


   2. Structure du Projet

Pediatric-Appendicitis-Diagnosis
│── application
│   ├── appendicitis_diagnosis_app.py   # Interface de l'application web
│   ├── force_plot.png                  # Visualisation SHAP (force plot)
│   ├── model_interpretation.py         # Analyse explicative basée sur SHAP
│
│── clean-data
│   ├── cleaning_data.ipynb              # Notebook de nettoyage des données
│
│── data                                 # Données brutes et nettoyées
│
│── Machine_learning
│   ├── Training.py                      # Script d'entraînement des modèles
│   ├── best_model.py                    # Modèle LightGBM optimisé
│
│── Optimizing Memory Usage
│   ├── data_processing.py               # Fonctions d'optimisation mémoire
│   ├── test_memory_optimization.py      # Tests d'optimisation mémoire
│
│── .gitignore
│── README.md.txt                        # Documentation



   3. Prétraitement des Données
Gestion des Valeurs Manquantes
✔ Les valeurs manquantes ont été remplies avec des techniques d’imputation appropriées (moyenne des données des patients de meme age et meme taille).
✔ Les colonnes ayant plus de 80 % de valeurs manquantes ont été supprimées pour garantir la fiabilité du modèle.
✔ Les caractéristiques redondantes (ex. Poids vs. IMC) ont été supprimées pour éviter la multi-colinéarité.

Gestion des Valeurs Aberrantes et des Corrélations
✔ Les valeurs aberrantes ont été détectées et traitées à l’aide de méthodes statistiques.
✔ Les caractéristiques fortement corrélées ont été identifiées et une seule de chaque paire corrélée a été conservée (exemple BMI = Height/Weight).


   4. Modèles de Machine Learning
Trois modèles ont été entraînés et évalués pour sélectionner le meilleur :

| Modèle                   | Précision | Recall  | F1-score | AUC-ROC |
|--------------------------|-----------|---------|----------|---------|
|     SVM                  | 0.9236    | 0.9236  | 0.9231   | 0.9795  |
|   Random Forest          | 0.9363    | 0.9363  | 0.9355   | 0.9939  |
| LightGBM(modèle choisi)  | 0.9618    | 0.9618  | 0.9618   | 0.9777  |


✔ LightGBM a obtenu les meilleures performances, ce qui en fait le modèle sélectionné pour le déploiement.


   5. Évaluation du Modèle & Analyse SHAP
Métriques de Performance
Le meilleur modèle a été choisi en fonction de :
✔ Précision, Recall et F1-score pour mesurer la qualité des prédictions.
✔ AUC-ROC pour évaluer la capacité du modèle à distinguer les cas positifs et négatifs.

SHAP (SHapley Additive exPlanations)
✔ L’analyse SHAP a permis d’interpréter les décisions du modèle.
✔ Les caractéristiques cliniques les plus influentes dans le diagnostic de l’appendicite sont :

Localisation de la douleur abdominale
Nombre de globules blancs (WBC)
Présence de fièvre
Sensibilité à la palpation (Rebond abdominal)
✔ Une visualisation en force plot a été générée pour illustrer l’impact de chaque caractéristique sur la prédiction.


   6. Application Web
Une interface intuitive a été développée avec Streamlit (ou Flask) pour permettre aux médecins de :
✔ Saisir les symptômes et résultats d’examens d’un patient.
✔ Obtenir une prédiction instantanée du risque d’appendicite.
✔ Comprendre la décision du modèle grâce à une analyse SHAP détaillée.

   7. Optimisation de la Mémoire & Prompt Engineering
✔ Des techniques de prompt engineering ont été appliquées pour améliorer le traitement des données et optimiser la mémoire.
✔ Le module data_processing.py a été conçu pour réduire la consommation mémoire lors du chargement des données médicales.
✔ Les types de données ont été convertis (float64 → float32, int64 → int32) pour minimiser l’empreinte mémoire sans perte de précision.

   8. Principaux Résultats
a. L'ensemble de données était-il équilibré ?
Oui, il était approximativement équilibré (~50 % appendicite, ~50 % non-appendicite).

b. Quel modèle de ML a donné les meilleures performances ?
LightGBM a obtenu les meilleurs scores en termes de précision et de robustesse.

c. Quelles caractéristiques cliniques influencent le plus le diagnostic ?
L’analyse SHAP a révélé que la localisation de la douleur, le WBC, la fièvre et la sensibilité abdominale sont les principaux indicateurs.

d. Apports du Prompt Engineering ?
Une approche structurée des prompts a permis d’améliorer l’efficacité des fonctions de traitement des données et de mémoire.


   9. Améliorations Futures
🔹 Enrichir le jeu de données avec des caractéristiques supplémentaires pour une meilleure généralisation.
🔹 Améliorer l’explicabilité du modèle pour faciliter son adoption par les professionnels de santé.
🔹 Déployer l’application web dans des hôpitaux et cliniques pour des tests en conditions réelles.