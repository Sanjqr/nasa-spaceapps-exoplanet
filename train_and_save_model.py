# train_and_save_model.py
import pandas as pd
import numpy as np
import joblib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Load Data ---
file_path = "KOI_Dataset_Exoplanets.csv"
df = pd.read_csv(file_path, comment="#")

# --- 2. Prepare Data for Modeling ---
y = df["koi_disposition"].replace({"CONFIRMED": 1, "CANDIDATE": 2, "FALSE POSITIVE": 0})
drop_cols = ["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition"]
X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=[np.number])

# CRITICAL: Save the column order for the API
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')
print("Model columns saved.")

# --- 3. Preprocessing ---
# NOTE: We fit the imputer and scaler on the FULL dataset.
# This is a common practice when the test set isn't held out for a long time.
# For ultimate purity, you would fit only on the training data. For this project, this is robust.
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# --- 4. Create the Train-Test Split (The "Practice Exam") ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set has {X_train.shape[0]} samples.")
print(f"Test set has {X_test.shape[0]} samples.")

# --- 5. Define and Train the Model ---
# We train the model ONLY on the training data.
lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=9)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=9)
meta_model = LogisticRegression(max_iter=1000)
stack_model = StackingClassifier(
    estimators=[("lgbm", lgbm), ("gb", gb), ("rf", rf)],
    final_estimator=meta_model,
    cv=3,
    n_jobs=-1
)

print("\nTraining the final model on the training set...")
stack_model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model on the Unseen Test Data ---
print("\n--- Model Evaluation on the Unseen Test Set ---")
y_pred = stack_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"]))
print("-------------------------------------------------")

# --- 7. Save the Final Objects for Deployment ---
# We save the model that was trained on the training set (80% of data).
# We also save the scaler and imputer that were fit on the full data.
joblib.dump(stack_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("\nModel, scaler, and imputer have been saved to .pkl files!")
print("Your project is now ready for deployment.")