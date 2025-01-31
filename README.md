Fake News Detection - Synthèse des Expériences avec Différents Modèles
🌟 Projet de Détection de Fake News
💪 Équipe : Mohamed Boumrar, Fred Mendes, Yannick Howaton

🌐 Contexte
Dans le cadre de ce projet, nous avons exploré plusieurs approches pour détecter les fake news à partir de données textuelles. Nous avons testé différentes techniques et architectures afin de comparer leur efficacité et optimiser les performances de classification.

🎯 Objectif
L'objectif global était de développer plusieurs modèles capables de détecter les fake news et d'analyser leurs performances respectives. Nous avons exploré trois approches principales :
1. TF-IDF + Régression Logistique avec Ponctuation : Amélioration de la vectorisation TF-IDF en conservant la ponctuation pour de meilleures performances.
2. BERT : Utilisation de BERT (Bidirectional Encoder Representations from Transformers) pour une meilleure compréhension du contexte global.
3. TF-IDF + Régression Logistique avec SMOTE : Approche basée sur la vectorisation TF-IDF et une régression logistique pour la classification.

________________________________________

📝 Modélisation et Expériences

💡 1. TF-IDF + Régression Logistique avec Ponctuation
Pourquoi cette approche ?
- L'inclusion de la ponctuation permet de capturer davantage de nuances dans le texte.
- TF-IDF permet d'extraire efficacement les caractéristiques textuelles.
- Régression logistique pour une classification rapide et performante.

Problèmes et solutions :
- Sensibilité à la qualité des textes → Prétraitement avancé et nettoyage optimisé.

📊 Résultats :
- Accuracy : 91%
- F1-score : 90% / 92%
- Bon compromis entre rapidité et précision.

________________________________________

🔍 2. BERT pour la détection de fake news
Pourquoi BERT ?
BERT est un modèle de transformer bidirectionnel pré-entraidné, extrêmement puissant pour le NLP.

Difficultés et solutions :
- Ressources lourdes → Utilisation de DistilBERT pour réduire la complexité.

📊 Résultats :
- Accuracy : 90% sur les données de test.
- Meilleures performances mais fortement coûteux.

________________________________________

🌟 3. TF-IDF + Régression Logistique avec SMOTE
Pourquoi cette approche ?
- TF-IDF pour extraire les caractéristiques textuelles.
- Régression logistique comme modèle simple et efficace.
- SMOTE pour corriger le déséquilibre des classes.

Problèmes et solutions :
- Déséquilibre des classes → Utilisation de SMOTE.

📊 Résultats :
- Accuracy : 95.81%
- F1-score : 95.75%
- Approche équilibrée entre performance et complexité.

________________________________________

📈 Conclusion Générale

| Modèle | Accuracy | Points forts | Limitations |
|---------|---------|--------------|--------------|
| TF-IDF + Régression Logistique (Ponctuation) | 91% | Bon compromis entre rapidité et précision | Sensible à la qualité des données |
| BERT | 90% | Précision très élevée | Exigeant en ressources |
| TF-IDF + Logistic Regression (SMOTE) | 95.81% | Bonne performance et gestion des classes | Moins adapté aux contextes complexes |

Synthèse :
- BERT est le plus précis mais très coûteux.
- TF-IDF + Régression Logistique (Ponctuation) est rapide et efficace.
- TF-IDF + Logistic Regression (SMOTE) est performant et bien équilibré.

👨‍💼 Contribution
Ce projet a été réalisé par Mohamed Boumrar, Fred Mendes et Yannick Howaton. Chaque modèle a été testé et optimisé en équipe pour assurer des performances solides.

🛠️ Installation et Utilisation
1. Voir le projet sur le lien suivant : https://github.com/FMendes13/Full_Stack_Project_DEEP_LEARNING_NLP
2. Charger le modèle et prédire un article :
```python
import pickle
with open('logistic_regression_model_smote.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
prediction = model.predict([article_text])
```
________________________________________

📁 Ce README fournit une vue d'ensemble claire et structurée du projet, mettant en valeur l'expérimentation et la comparaison des modèles.
