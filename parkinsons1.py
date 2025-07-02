print("Script started")

# === Imports ===
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings("ignore")

print("Imports done")

# === Load Dataset ===
df = pd.read_csv("parkinsons.data")
df.drop(columns=['name'], inplace=True)  # Drop non-numeric column
print(f"Loaded dataset with shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# === Split Features & Labels ===
X = df.drop('status', axis=1)
y = df['status']

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled")

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
print("Data split into train and test")

# === Handle Class Imbalance ===
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Applied SMOTE to balance classes")

# === Genetic Algorithm for Feature Selection ===
print("Running FAST Genetic Algorithm for feature selection...")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def evalFitness(individual):
    selected = [i for i in range(len(individual)) if individual[i] == 1]
    if len(selected) == 0:
        return 0.0,
    X_sel = X_train[:, selected]
    clf = LogisticRegression(max_iter=1000)
    score = np.mean(cross_val_score(clf, X_sel, y_train, cv=5))
    return score,

toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=30)
hof = tools.HallOfFame(1)

algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=15, halloffame=hof, verbose=True)

best_individual = hof[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
print(f"Genetic Algorithm complete\nBest feature subset selected: {selected_features}")

X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]

# === Model Tuning and Training ===
print("Tuning classifiers...")

# Define base learners
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42))
]

# Meta-learner
final_estimator = LogisticRegression(max_iter=1000)

# Stacking Classifier
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)

print("Training stacking classifier...")
stacking_clf.fit(X_train_sel, y_train)
print("Stacking classifier trained")

# === Evaluate Model ===
print("Evaluating model...\n")
y_pred = stacking_clf.predict(X_test_sel)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Script finished")


