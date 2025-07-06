import pandas as pd  # Pour manipuler les données (tableaux, colonnes, etc.)
import matplotlib.pyplot as plt  # Pour faire des graphiques
import seaborn as sns  # Pour les graphiques stylés (genre heatmap)
import joblib
import os



# Librairies Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# === 1. Chargement des données ===
df = pd.read_csv('data/creditcard.csv') 

# === 2. Prétraitement (mise à l'échelle) ===
# On scale 'Amount' pour enlever les valeurs extrêmes (fraude = souvent montants bizarres)
scaler = RobustScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].to_numpy().reshape(-1, 1))

# On normalise 'Time' entre 0 et 1 pour pas qu'elle pèse trop dans les modèles
df['Time'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())

# === 3. Séparation des données ===
X = df.drop('Class', axis=1)  # toutes les colonnes sauf 'Class'
y = df['Class']  # la colonne à prédire : 0 = normal, 1 = fraude

# On sépare en entraînement/test (80/20), tout en gardant le ratio fraude/pas fraude (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# === 4. Entraînement des modèles ===

# Modèle simple : Régression logistique
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Modèle un peu plus puissant : Random Forest
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# === 5. Évaluation des performances ===
print("\n=== Régression Logistique ===")
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# === 6. Matrices de confusion ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix - Logistic Regression")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("models/confusion_matrices.png")  # Sauvegarde au lieu d'afficher
plt.close()
#plt.show()

# === 7. Courbe ROC pour Random Forest ===
# Elle permet de visualiser le compromis entre détection correcte et faux positifs
y_scores_rf = model_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores_rf)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_scores_rf):.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Random Forest")
plt.legend()
plt.grid()
plt.savefig("models/roc_curve.png")
plt.close()
#plt.show(block=False)
#plt.pause(5)
#plt.close()

# Sauvegarder le modèle Random Forest
joblib.dump(model_rf, 'models/random_forest.joblib')
