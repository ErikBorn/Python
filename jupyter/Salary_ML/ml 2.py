import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import cross_validate
from matplotlib import pyplot as plt

# Load updated dataset
df = pd.read_csv("/Users/erikborn/Documents/Python/jupyter/Salary_ML/modelData.csv")

# Preprocessing
df["Hire Date "] = pd.to_datetime(df["Hire Date "], errors="coerce")
df["years_since_hire"] = 2025 - df["Hire Date "].dt.year

df["Ethnicity"] = df["Ethnicity"].fillna("Unknown")

# Optional cleanup: convert salary strings like "$67,500" to numbers
df["25-26 Salary"] = pd.to_numeric(df["25-26 Salary"].astype(str).str.replace("[$,]", "", regex=True), errors="coerce")

# Drop rows with missing target
df = df.dropna(subset=["25-26 Salary"])

# Select updated features
features = [
    "Gender",
    "Ethnicity",
    "years_since_hire",
    "Years of Exp",
    "Education Level",
    "Skill Rating",
    "Knowledge Rating",
    "Prep Rating",
    "Seniority"
]

categorical_features = [
    "Gender",
    "Ethnicity",
    "Education Level"
]

X = df[features].copy()

X["Ethnicity"] = df["Ethnicity"].astype("category")

# Ensure categorical fields are typed correctly
for cat_col in categorical_features:
    X[cat_col] = X[cat_col].astype("category")

y = df["25-26 Salary"].copy()

# Train model
model = xgb.XGBRegressor(
    max_depth=2,
    learning_rate=0.05,     # Slower learning
    n_estimators=200,       # More trees, but gentler updates
    subsample=0.7,          # Less chance of overfitting
    colsample_bytree=0.8,   # Only use 80% of features per tree
    enable_categorical=True,
    max_cat_to_onehot=1,
    reg_lambda=10,          # Stronger regularization
    reg_alpha=5
)

# Cross-validated scores
scores = cross_validate(
    model,
    X,
    y,
    cv=5,
    scoring=('r2', 'neg_mean_squared_error'),
    return_train_score=True,
)

print(pd.DataFrame(scores))

# Fit full model for SHAP interpretation
model.fit(X, y)
y_pred = model.predict(X)

# SHAP values and beeswarm plot
explainer = shap.Explainer(model)
shap_values = explainer(X)
# shap.summary_plot(shap_values, X, plot_type="bar")
shap.plots.beeswarm(shap_values, max_display=30)