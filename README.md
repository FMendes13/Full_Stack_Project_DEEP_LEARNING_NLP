# Fake News Detection with TF-IDF and Logistic Regression

## Projet de Détection de Fake News

**Equipe**: Mohamed Boumrar, Fred Mendes, Yannick Howaton

**Contexte**:  
Dans le cadre de notre projet, nous avons développé un modèle de détection de fake news en utilisant la méthode TF-IDF combinée à une régression logistique. Ce projet a pour objectif d'analyser et de classifier des articles de presse en fonction de leur véracité, en se basant sur un ensemble de données étiqueté.

## Objectifs

L'objectif principal est de créer un modèle performant capable de distinguer les fake news des articles réels à partir de données textuelles. Nous avons utilisé un ensemble de données avec des articles étiquetés comme étant "vrai" ou "faux". Ce projet implique des étapes d'exploration, de prétraitement des données, d'entraînement de modèles, et d'évaluation de leur performance.

## Installation

 1. Clonez le dépôt :  
   ```bash
   git clone https://github.com/Boumrarmohamed/fake_news_tfidf.git


## Description du Projet
# Prétraitement des Données
Les étapes de prétraitement ont été réalisées avec la bibliothèque pandas pour nettoyer et structurer les données avant de les utiliser pour l'entraînement du modèle. Cela inclut :

Le nettoyage des données textuelles
La suppression des doublons
Le traitement des valeurs manquantes
Le filtrage des stop words (mots vides) grâce à un fichier stop_words.txt

# Modélisation
Nous avons utilisé la méthode TF-IDF pour vectoriser les textes et la régression logistique comme modèle de classification. Le modèle a été entraîné sur les données d'entraînement, et la performance a été évaluée à l'aide de la précision, du rappel, du score F1, et de la matrice de confusion.

# Techniques avancées
SMOTE (Synthetic Minority Over-sampling Technique) a été appliqué pour résoudre le problème de déséquilibre des classes.
Une recherche de grille (GridSearch) a été effectuée pour optimiser les hyperparamètres du modèle.

# Évaluation
Les performances du modèle ont été mesurées avec les métriques suivantes :

Précision: 0.96
Rappel: 0.96
F1-score: 0.96
Accuracy: 95.8%
Une matrice de confusion a été générée pour visualiser les résultats et analyser les faux positifs et faux négatifs.

## Fichiers du Projet
logistic_regression_model_smote.pkl : modèle entraîné avec SMOTE
matrice.png : image de la matrice de confusion
output.png : graphique des résultats d'évaluation
stop_words.txt : fichier contenant les mots vides (stop words) utilisés pour le prétraitement

## Utilisation
Pour utiliser le modèle, il suffit de charger le fichier logistic_regression_model_smote.pkl et d'utiliser la méthode predict pour classifier de nouveaux articles.

Exemple :


import pickle

# Charger le modèle
with open('logistic_regression_model_smote.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Prédire avec un article
prediction = model.predict([article_text])


## Contributions
Ce projet a été réalisé en collaboration avec mes collègues Fred Mendes et Yannick Howaton. Chaque étape de développement a été partagée, testée et améliorée en équipe pour garantir la qualité et la robustesse du modèle.
