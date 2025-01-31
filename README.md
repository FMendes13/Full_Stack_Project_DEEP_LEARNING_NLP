Fake News Detection 
Synthèse des Expériences avec Différents Modèles

🌟 Projet de Détection de Fake News
💪 Équipe : Mohamed Boumrar, Fred Mendes, Yannick Howaton

🌐 Contexte
Dans le cadre de ce projet, nous avons exploré plusieurs approches pour détecter les fake news à partir de données textuelles. Nous avons testé différentes techniques et architectures afin de comparer leur efficacité et optimiser les performances de classification.

🎯 Objectif
L'objectif global était de développer plusieurs modèles capables de détecter les fake news et d'analyser leurs performances respectives. Nous avons exploré trois approches principales :
TF-IDF + Régression Logistique avec et sans Ponctuation : Amélioration de la vectorisation TF-IDF en conservant la ponctuation pour de meilleures performances.
3. TF-IDF + Régression Logistique avec SMOTE : Approche basée sur la vectorisation TF-IDF et une régression logistique pour la classification.
2. BERT : Utilisation de BERT (Bidirectional Encoder Representations from Transformers) pour une meilleure compréhension du contexte global.

________________________________________

📝 Modélisation et Expériences

💡 1. TF-IDF + Régression Logistique avec et sans Ponctuation
Pourquoi cette approche ?
- L'inclusion de la ponctuation permettrait de capturer davantage de nuances dans le texte.
- TF-IDF permet d'extraire efficacement les caractéristiques textuelles.
- Régression logistique pour une classification rapide et performante.

Problèmes et solutions :
- Sensibilité à la qualité des textes → Prétraitement avancé et nettoyage optimisé.

📊 Résultats :
- Accuracy : 91%
- F1-score : 90% / 92%
- Bon compromis entre rapidité et précision.
————————
🌟 2. TF-IDF + Régression Logistique avec SMOTE
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

🔍 3. BERT pour la détection de fake news
Pourquoi BERT ?
BERT est un modèle de transformer bidirectionnel pré-entraidné, extrêmement puissant pour le NLP.

Difficultés et solutions :
- Ressources lourdes → Utilisation de DistilBERT pour réduire la complexité.

📊 Résultats :
- Accuracy : 90% sur les données de test.
- Meilleures performances mais fortement coûteux.


________________________________________

📈 Conclusion Générale

| Modèle | Accuracy | Points forts | Limitations |
|---------|---------|--------------|--------------|
| BERT | 90% | Précision très élevée | Exigeant en ressources |
| TF-IDF + Régression Logistique (Ponctuation) | 91% | Bon compromis entre rapidité et précision | Sensible à la qualité des données |
| TF-IDF + Logistic Regression (SMOTE) | 95.81% | Bonne performance et gestion des classes | Moins adapté aux contextes complexes |

Synthèse :
- BERT est le plus précis mais très coûteux.
- TF-IDF + Régression Logistique (Ponctuation) est rapide et efficace.
- TF-IDF + Logistic Regression (SMOTE) est performant et bien équilibré.

🛠️ Installation et Utilisation:
Chargez le modèle et prédire un article :
```python
import pickle
with open('logistic_regression_model_smote.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
prediction = model.predict([article_text])
```
________________________________________
