import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("FAOSTAT1.csv")


# ============================================================
# CLEAN AND RESHAPE DATA
# ============================================================

df = df[df["Element"].isin([
    "Area harvested",
    "Production",
    "Yield"
])].copy()

df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

df = df.pivot_table(
    index="Year",
    columns="Element",
    values="Value",
    aggfunc="mean"
).reset_index()

df = df.rename(columns={
    "Area harvested": "Area_Harvested",
    "Production": "Production",
    "Yield": "Yield"
})

df = df.dropna()
df = df.sort_values("Year").reset_index(drop=True)


# ============================================================
# 80% TRAINING - 20% TESTING
# ============================================================

split = int(len(df) * 0.80)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["Year"]]
X_test = test[["Year"]]

y_train = train["Production"]
y_test = test["Production"]


# ============================================================
# POLYNOMIAL REGRESSION
# Degree = 2
# ============================================================

model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X_train, y_train)

prediction = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)


# ============================================================
# BEAUTIFUL OUTPUT BOX
# ============================================================

print("\n")

print("╔══════════════════════════════════════════════════════════╗")
print("║           POLYNOMIAL REGRESSION RESULTS                  ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  Polynomial Degree : {2:<34}║")
print(f"║  Training Data     : {len(train):<34}║")
print(f"║  Testing Data      : {len(test):<34}║")
print(
    f"║  Year Range        : "
    f"{int(df['Year'].min())} - {int(df['Year'].max()):<24}║"
)

print("╠══════════════════════════════════════════════════════════╣")
print("║                    MODEL METRICS                         ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  MSE      : {mse:,.2f}".ljust(59) + "║")
print(f"║  RMSE     : {rmse:,.2f}".ljust(59) + "║")
print(f"║  MAE      : {mae:,.2f}".ljust(59) + "║")
print(f"║  R² Score : {r2:.4f}".ljust(59) + "║")

print("╠══════════════════════════════════════════════════════════╣")
print("║                    MODEL INFORMATION                    ║")
print("╠══════════════════════════════════════════════════════════╣")

print("║  Input Variable : Year                                  ║")
print("║  Target Variable: Production                            ║")
print("║  Model Type     : Polynomial Regression                  ║")
print("║  Degree         : 2                                     ║")

print("╚══════════════════════════════════════════════════════════╝")


# ============================================================
# ACTUAL VS PREDICTED TABLE
# ============================================================

print("\n")
print("┌──────────┬────────────────────┬────────────────────┐")
print("│   Year   │   Actual Production │ Predicted Production│")
print("├──────────┼────────────────────┼────────────────────┤")

for year, actual, pred in zip(
    test["Year"],
    y_test,
    prediction
):
    print(
        f"│ {int(year):^8} │ "
        f"{actual:>18,.0f} │ "
        f"{pred:>18,.0f} │"
    )

print("└──────────┴────────────────────┴────────────────────┘")


# ============================================================
# GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    test["Year"],
    y_test,
    label="Actual Production"
)

plt.plot(
    test["Year"],
    prediction,
    label="Polynomial Prediction"
)

plt.xlabel("Year")
plt.ylabel("Production")

plt.title(
    "Polynomial Regression - Wheat Production"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()