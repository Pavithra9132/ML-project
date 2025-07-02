print("Script started")

# ===================== Imports =====================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

print("Imports done")

# ===================== 1. Load and Clean Dataset =====================
data = pd.read_csv('parkinsons1.data')
data = data[data['age'] != 'age']  # Remove duplicate header if present
data = data.astype(float)
print(f"Loaded dataset with shape: {data.shape}")
print("Column names:", data.columns.tolist())

# ===================== 2. Prepare Features =====================
X = data.drop(columns=['subject#', 'total_UPDRS', 'motor_UPDRS'])
y = data['total_UPDRS']

# ===================== 3. Scale Features =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled")

# ===================== 4. Train-Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data split into train and test")

# ===================== 5. FAST Genetic Algorithm =====================
from sklearn.linear_model import LinearRegression  # faster model for GA

n_features = X.shape[1]
cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(individual):
    if sum(individual) == 0:
        return 9999.,
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = X_train[:, selected]
    model = LinearRegression()
    scores = -cross_val_score(model, X_sel, y_train, cv=cv_strategy,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
    return scores.mean(),

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

print("Running FAST Genetic Algorithm for feature selection...")
population = toolbox.population(n=10)  # reduced from 30
hof = tools.HallOfFame(1)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3,
                    ngen=10, halloffame=hof, verbose=True)  # reduced from 50
print("Genetic Algorithm complete")

best_ind = hof[0]
selected_features = [i for i, bit in enumerate(best_ind) if bit == 1]
print(f"Best feature subset selected: {selected_features}")

X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]

# ===================== 6. Hyperparameter Tuning =====================
print("Starting hyperparameter tuning for RandomForest...")
rf = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf, {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}, n_iter=4, cv=cv_strategy, scoring='neg_root_mean_squared_error', random_state=42)
rf_search.fit(X_train_sel, y_train)
print("RandomForest tuning done")

print("Starting hyperparameter tuning for XGBoost...")
xgb = XGBRegressor(random_state=42)
xgb_search = RandomizedSearchCV(xgb, {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}, n_iter=4, cv=cv_strategy, scoring='neg_root_mean_squared_error', random_state=42)
xgb_search.fit(X_train_sel, y_train)
print("XGBoost tuning done")

print("Starting hyperparameter tuning for GradientBoosting...")
gb = GradientBoostingRegressor(random_state=42)
gb_search = RandomizedSearchCV(gb, {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1]
}, n_iter=4, cv=cv_strategy, scoring='neg_root_mean_squared_error', random_state=42)
gb_search.fit(X_train_sel, y_train)
print("GradientBoosting tuning done")

print("Starting hyperparameter tuning for SVR...")
svr = SVR()
svr_search = RandomizedSearchCV(svr, {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}, n_iter=4, cv=cv_strategy, scoring='neg_root_mean_squared_error', random_state=42)
svr_search.fit(X_train_sel, y_train)
print("SVR tuning done")

# ===================== 7. Train Stacking Regressor =====================
print("Training stacking regressor...")
stack_model = StackingRegressor(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('xgb', xgb_search.best_estimator_),
        ('gb', gb_search.best_estimator_),
        ('svr', svr_search.best_estimator_)
    ],
    final_estimator=LinearRegression(),
    passthrough=True,
    cv=cv_strategy
)
stack_model.fit(X_train_sel, y_train)
print("Stacking regressor trained")

# ===================== 8. Evaluate Model =====================
print("Predicting on test set...")
y_pred = stack_model.predict(X_test_sel)

print("Evaluating model...")
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Stacking RMSE: {rmse:.4f}")
print(f"✅ Stacking R² Score: {r2:.4f}")
print("✅ Script finished")
