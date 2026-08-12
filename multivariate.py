import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
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


# ============================================================
# MULTIVARIATE VARIABLES
# ============================================================

X_train = train[
    ["Year", "Area_Harvested", "Yield"]
]

X_test = test[
    ["Year", "Area_Harvested", "Yield"]
]

y_train = train["Production"]
y_test = test["Production"]


# ============================================================
# MULTIVARIATE LINEAR REGRESSION
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

mse = mean_squared_error(
    y_test,
    prediction
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    prediction
)

r2 = r2_score(
    y_test,
    prediction
)


# ============================================================
# BEAUTIFUL OUTPUT BOX
# ============================================================

print("\n")

print("╔══════════════════════════════════════════════════════════╗")
print("║          MULTIVARIATE REGRESSION RESULTS                 ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  Training Data : {len(train):<37}║")
print(f"║  Testing Data  : {len(test):<37}║")

print(
    f"║  Year Range    : "
    f"{int(df['Year'].min())} - {int(df['Year'].max()):<27}║"
)

print("╠══════════════════════════════════════════════════════════╣")
print("║                  INPUT VARIABLES                         ║")
print("╠══════════════════════════════════════════════════════════╣")

print("║  1. Year                                                 ║")
print("║  2. Area Harvested                                       ║")
print("║  3. Yield                                                ║")

print("║                                                          ║")
print("║  Target: Production                                      ║")

print("╠══════════════════════════════════════════════════════════╣")
print("║                    MODEL METRICS                         ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  MSE      : {mse:,.2f}".ljust(59) + "║")
print(f"║  RMSE     : {rmse:,.2f}".ljust(59) + "║")
print(f"║  MAE      : {mae:,.2f}".ljust(59) + "║")
print(f"║  R² Score : {r2:.4f}".ljust(59) + "║")

print("╠══════════════════════════════════════════════════════════╣")
print("║                    COEFFICIENTS                          ║")
print("╠══════════════════════════════════════════════════════════╣")

print(
    f"║  Year            : {model.coef_[0]:,.4f}".ljust(59) + "║"
)

print(
    f"║  Area Harvested  : {model.coef_[1]:,.4f}".ljust(59) + "║"
)

print(
    f"║  Yield           : {model.coef_[2]:,.4f}".ljust(59) + "║"
)

print(
    f"║  Intercept       : {model.intercept_:,.4f}".ljust(59) + "║"
)

print("╚══════════════════════════════════════════════════════════╝")


# ============================================================
# EQUATION
# ============================================================

print("\n")
print("╔══════════════════════════════════════════════════════════╗")
print("║                  MODEL EQUATION                          ║")
print("╠══════════════════════════════════════════════════════════╣")

print(
    f"║ Production = {model.intercept_:,.2f}"
)

print(
    f"║ + ({model.coef_[0]:,.2f} × Year)"
)

print(
    f"║ + ({model.coef_[1]:,.2f} × Area Harvested)"
)

print(
    f"║ + ({model.coef_[2]:,.2f} × Yield)"
)

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
    label="Multivariate Prediction"
)

plt.xlabel("Year")
plt.ylabel("Production")

plt.title(
    "Multivariate Regression - Wheat Production"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()