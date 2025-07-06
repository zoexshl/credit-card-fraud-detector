# Détecteur de Fraude par Carte de Crédit

> **Un système intelligent de détection de fraudes utilisant Machine Learning et une interface Streamlit interactive**

---

## **Aperçu du Projet**

Ce projet implémente un **système de détection de fraudes** en temps réel pour les transactions par carte de crédit. Il combine des techniques avancées de Machine Learning avec une interface utilisateur moderne pour identifier automatiquement les transactions suspectes.

### **Caractéristiques Principales**

- **Détection en temps réel** : Analyse instantanée des transactions
- **Haute précision** : Modèles optimisés pour minimiser les faux positifs
- **Interface interactive** : Dashboard complet avec visualisations
- **Facilement déployable** : Interface Streamlit prête à l'emploi
- **Métriques détaillées** : Rapport de performance complet

---

## **Architecture du Système**

<img src="https://github.com/user-attachments/assets/3a3aaf18-5278-46fd-a83a-4cad4eb9c7a5"/>

## **Installation et Configuration**

### Prérequis
```
streamlit
pandas
scikit-learn
matplotlib
seaborn
plotly
joblib
numpy
```

### Structure du projet
```
fraud-detection/
├── README.md                 # Documentation
├── credit.py                 # Script d'entraînement
├── app.py                    # Interface Streamlit
├── data/
│   └── creditcard.csv          # Dataset (non inclus)
└──  models/
    ├── random_forest.joblib    # Modèle entraîné
    ├── confusion_matrices.png  # Matrices de confusion
    └── roc_curve.png           # Courbe ROC

```

---

## **Méthodologie Scientifique**

### 1. **Préparation des Données**
- **Normalisation** : Scaling robuste pour les montants
- **Transformation temporelle** : Normalisation des timestamps
- **Gestion du déséquilibre** : Stratification pour préserver les ratios

### 2. **Modèles Implémentés**

| Modèle | Avantages | Cas d'usage |
|--------|-----------|-------------|
| **Régression Logistique** |  Rapide, interprétable | Baseline, production légère |
| **Random Forest** | Haute précision, robuste | Production recommandée |

### 3. **Métriques d'Évaluation**

```python
Précision = TP / (TP + FP)    # Éviter les faux positifs
Rappel = TP / (TP + FN)       # Capturer toutes les fraudes
F1-Score = 2 * (P * R) / (P + R)  # Équilibre optimal
```

---

## **Résultats et Performance**

### Comparaison des performances
<img src="https://github.com/user-attachments/assets/8ff5e503-cda8-4b65-a4c7-a1b09af17932"/>

### Courbe ROC
<img src="https://github.com/user-attachments/assets/d6ebbd6e-1043-4e82-abd2-a5ec84708124" width="600"/>

### **Performances du Modèle Random Forest**
- **Précision** : 92.6% ✨
- **Rappel** : 70.7% ✨ 
- **F1-Score** : 80.4% ✨ 
- **AUC-ROC** : 95.7% ✨ 

---

## **Interface Utilisateur**

### Dashboard Principal
<img src="https://github.com/user-attachments/assets/136113d9-ff30-43c3-a170-a8089bd5c2a6"/>

### Analyse des Prédictions
<img src="https://github.com/user-attachments/assets/338bb073-8929-4136-b7bb-b41e83aac957"/>

---

## **Utilisation**

### 1. Entraînement du modèle
```bash
python credit.py
```
**Sortie** :
- Modèle sauvegardé dans `models/`
- Graphiques de performance générés
- Métriques affichées en console

### 2. Lancement de l'interface
```bash
streamlit run app.py
```

### 3. Analyse de transactions
1. **Upload** : Déposez votre fichier CSV
2. **Analyse** : Le modèle traite automatiquement les données
3. **Résultats** : Consultez les onglets pour l'analyse complète
4. **Export** : Téléchargez le rapport final

---

## **Fonctionnalités Avancées**

### **Onglet Aperçu**
- Statistiques descriptives instantanées
- Validation automatique du format des données
- Aperçu des premières transactions

### **Onglet Prédictions**
- Détection automatique des fraudes
- Filtrage par type de transaction
- Probabilités de fraude individuelles
- Export des résultats

### **Onglet Visualisations**
- Distribution des prédictions
- Analyse des montants par catégorie
- Histogrammes de probabilités
- Graphiques interactifs Plotly

### **Onglet Rapport**
- Métriques de performance détaillées
- Matrice de confusion interactive
- Analyse comparative si labels disponibles
- Résumé exécutif

---
### API du Modèle

| Méthode | Description | Paramètres | Retour |
|---------|-------------|------------|---------|
| `predict()` | Prédiction binaire | `X: array` | `[0,1]` |
| `predict_proba()` | Probabilités | `X: array` | `[[p0,p1]]` |
| `score()` | Accuracy | `X, y` | `float` |


---

## **Crédits**

### Dataset
Dataset utilisé : [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) - Kaggle

### Mentions
- **Scikit-Learn** : Framework ML
- **Streamlit** : Interface utilisateur
- **Plotly** : Visualisations interactives
- **Communauté Kaggle** : Dataset et inspiration


---

<div align="center">

![Fraude Detection](https://img.shields.io/badge/Fraud-Detection-red?style=for-the-badge&logo=shield&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue?style=for-the-badge&logo=brain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-green?style=for-the-badge&logo=streamlit&logoColor=white)

</div>
