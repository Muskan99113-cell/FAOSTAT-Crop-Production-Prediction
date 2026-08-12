import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load data
df = pd.read_csv("FAOSTAT1.csv")

# Filter required elements
df = df[df["Element"].isin([
    "Area harvested",
    "Production",
    "Yield"
])].copy()

df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Reshape
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


# 80/20 split
split = int(len(df) * 0.80)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[["Year"]]
X_test = test[["Year"]]

y_train = train["Production"]
y_test = test["Production"]


# Model
model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)


# Metrics
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)


# ============================================================
# BEAUTIFUL OUTPUT
# ============================================================

print("\n")
print("╔══════════════════════════════════════════════════════════╗")
print("║             LINEAR REGRESSION RESULTS                    ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  Training Data : {len(train):<37}║")
print(f"║  Testing Data  : {len(test):<37}║")
print(
    f"║  Year Range    : "
    f"{int(df['Year'].min())} - {int(df['Year'].max()):<27}║"
)

print("╠══════════════════════════════════════════════════════════╣")
print("║                    MODEL METRICS                         ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"║  MSE      : {mse:,.2f}".ljust(59) + "║")
print(f"║  RMSE     : {rmse:,.2f}".ljust(59) + "║")
print(f"║  MAE      : {mae:,.2f}".ljust(59) + "║")
print(f"║  R² Score : {r2:.4f}".ljust(59) + "║")

print("╠══════════════════════════════════════════════════════════╣")
print("║                    MODEL EQUATION                        ║")
print("╠══════════════════════════════════════════════════════════╣")

equation = (
    f"Production = {model.coef_[0]:,.2f} × Year "
    f"{model.intercept_:+,.2f}"
)

print(f"║  {equation}".ljust(59) + "║")

print("╚══════════════════════════════════════════════════════════╝")


# ============================================================
# PREDICTION TABLE
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
    label="Linear Prediction"
)

plt.xlabel("Year")
plt.ylabel("Production")
plt.title("Linear Regression - Wheat Production")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()