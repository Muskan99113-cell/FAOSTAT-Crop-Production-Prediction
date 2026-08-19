import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("FAOSTAT1.csv")


# ============================================================
# CLEAN DATA
# ============================================================

df = df[df["Element"].isin([
    "Area harvested",
    "Production",
    "Yield"
])].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

df = df.pivot_table(
    index="Year",
    columns="Element",
    values="Value",
    aggfunc="mean"
).reset_index()

df = df.rename(columns={
    "Area harvested": "Area_Harvested"
})

df = df.dropna()
df = df.sort_values("Year").reset_index(drop=True)


# ============================================================
# 80% TRAINING - 20% TESTING
# ============================================================

split = int(len(df) * 0.80)

train = df.iloc[:split]
test = df.iloc[split:]

y_train = train["Production"]
y_test = test["Production"]


# ============================================================
# LINEAR REGRESSION
# ============================================================

X_train_linear = train[["Year"]]
X_test_linear = test[["Year"]]

linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)

linear_pred = linear_model.predict(
    X_test_linear
)


# ============================================================
# POLYNOMIAL REGRESSION
# ============================================================

poly_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

poly_model.fit(
    X_train_linear,
    y_train
)

poly_pred = poly_model.predict(
    X_test_linear
)


# ============================================================
# MULTIVARIATE REGRESSION
# ============================================================

X_train_multi = train[
    ["Year", "Area_Harvested", "Yield"]
]

X_test_multi = test[
    ["Year", "Area_Harvested", "Yield"]
]

multi_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

multi_model.fit(
    X_train_multi,
    y_train
)

multi_pred = multi_model.predict(
    X_test_multi
)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return mse, rmse, mae, r2


linear_metrics = calculate_metrics(
    y_test,
    linear_pred
)

poly_metrics = calculate_metrics(
    y_test,
    poly_pred
)

multi_metrics = calculate_metrics(
    y_test,
    multi_pred
)


# ============================================================
# COMPARISON DATAFRAME
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Multivariate Regression"
    ],

    "MSE": [
        linear_metrics[0],
        poly_metrics[0],
        multi_metrics[0]
    ],

    "RMSE": [
        linear_metrics[1],
        poly_metrics[1],
        multi_metrics[1]
    ],

    "MAE": [
        linear_metrics[2],
        poly_metrics[2],
        multi_metrics[2]
    ],

    "R2": [
        linear_metrics[3],
        poly_metrics[3],
        multi_metrics[3]
    ]
})


# ============================================================
# TERMINAL HEADER
# ============================================================

print("\n")
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║                                                                      ║")
print("║                 REGRESSION MODEL COMPARISON                         ║")
print("║                                                                      ║")
print("╚══════════════════════════════════════════════════════════════════════╝")


# ============================================================
# DATASET INFORMATION BOX
# ============================================================

print()
print("┌──────────────────────────────────────────────────────────────────────┐")
print("│                         DATASET INFORMATION                          │")
print("├──────────────────────────────────────────────────────────────────────┤")

print(f"│  Crop              : Wheat                                           │")
print(f"│  Total Observations: {len(df):<48}│")
print(f"│  Training Data     : {len(train):<48}│")
print(f"│  Testing Data      : {len(test):<48}│")
print(
    f"│  Training Period   : "
    f"{int(train['Year'].min())} - {int(train['Year'].max()):<38}│"
)
print(
    f"│  Testing Period    : "
    f"{int(test['Year'].min())} - {int(test['Year'].max()):<38}│"
)
print("│  Train/Test Split  : 80% / 20%                                      │")

print("└──────────────────────────────────────────────────────────────────────┘")


# ============================================================
# MODEL PERFORMANCE BOX
# ============================================================

print()
print("┌──────────────────────────────────────────────────────────────────────┐")
print("│                       MODEL PERFORMANCE                              │")
print("├──────────────────────────────────────────────────────────────────────┤")

print(
    f"│ {'MODEL':<26}"
    f" {'RMSE':>14}"
    f" {'MAE':>14}"
    f" {'R²':>10} │"
)

print("├──────────────────────────────────────────────────────────────────────┤")

for _, row in comparison.iterrows():

    print(
        f"│ {row['Model']:<26}"
        f" {row['RMSE']:>14,.2f}"
        f" {row['MAE']:>14,.2f}"
        f" {row['R2']:>10.4f} │"
    )

print("└──────────────────────────────────────────────────────────────────────┘")


# ============================================================
# MSE BOX
# ============================================================

print()
print("┌──────────────────────────────────────────────────────────────────────┐")
print("│                         MSE RESULTS                                  │")
print("├──────────────────────────────────────────────────────────────────────┤")

for _, row in comparison.iterrows():

    print(
        f"│  {row['Model']:<28} : {row['MSE']:>25,.2f} │"
    )

print("└──────────────────────────────────────────────────────────────────────┘")


# ============================================================
# BEST MODEL BOX
# ============================================================

best_r2 = comparison.loc[
    comparison["R2"].idxmax()
]

best_rmse = comparison.loc[
    comparison["RMSE"].idxmin()
]

print()
print("┌──────────────────────────────────────────────────────────────────────┐")
print("│                         BEST MODEL                                   │")
print("├──────────────────────────────────────────────────────────────────────┤")

print(
    f"│  Highest R² Score : {best_r2['Model']:<42}│"
)

print(
    f"│  R² Score         : {best_r2['R2']:<42.4f}│"
)

print(
    f"│  Lowest RMSE      : {best_rmse['Model']:<42}│"
)

print(
    f"│  RMSE             : {best_rmse['RMSE']:<42,.2f}│"
)

print("└──────────────────────────────────────────────────────────────────────┘")


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({

    "Year": test["Year"].astype(int),

    "Actual": y_test.values,

    "Linear": linear_pred,

    "Polynomial": poly_pred,

    "Multivariate": multi_pred
})


print()
print("┌───────────────────────────────────────────────────────────────────────────────┐")
print("│                         ACTUAL VS PREDICTED                                   │")
print("├────────┬────────────────┬────────────────┬────────────────┬──────────────────┤")
print(
    f"│ {'YEAR':^6} "
    f"│ {'ACTUAL':^14} "
    f"│ {'LINEAR':^14} "
    f"│ {'POLYNOMIAL':^14} "
    f"│ {'MULTIVARIATE':^16} │"
)
print("├────────┼────────────────┼────────────────┼────────────────┼──────────────────┤")

for _, row in results.iterrows():

    print(
        f"│ {int(row['Year']):^6} "
        f"│ {row['Actual']:>14,.0f} "
        f"│ {row['Linear']:>14,.0f} "
        f"│ {row['Polynomial']:>14,.0f} "
        f"│ {row['Multivariate']:>16,.0f} │"
    )

print("└────────┴────────────────┴────────────────┴────────────────┴──────────────────┘")


# ============================================================
# GRAPH 1 - R2 COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.title(
    "R² Score Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Regression Model")
plt.ylabel("R² Score")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(bars, comparison["R2"]):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 2 - RMSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.title(
    "RMSE Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Regression Model")
plt.ylabel("RMSE")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    comparison["RMSE"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value / 1000000:.2f}M",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 3 - MAE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.title(
    "MAE Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Regression Model")
plt.ylabel("MAE")

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar, value in zip(
    bars,
    comparison["MAE"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value / 1000000:.2f}M",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 4 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    test["Year"],
    y_test,
    marker="o",
    label="Actual Production"
)

plt.plot(
    test["Year"],
    linear_pred,
    marker="o",
    label="Linear Regression"
)

plt.plot(
    test["Year"],
    poly_pred,
    marker="o",
    label="Polynomial Regression"
)

plt.plot(
    test["Year"],
    multi_pred,
    marker="o",
    label="Multivariate Regression"
)

plt.title(
    "Actual vs Predicted Wheat Production",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Year")
plt.ylabel("Production")

plt.legend()
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()