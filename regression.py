from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Fetch dataset
communities_and_crime = fetch_ucirepo(id=183)

# Get features and target
X = communities_and_crime.data.features.copy()
y = communities_and_crime.data.targets.squeeze()
alpha = 0.1
num = X.shape[0] #1984

X = X.drop(["state", "county", "community", "communityname", "fold"],axis=1)
X = X.replace("?", float("nan"))
X = X.astype(float)

imputer = SimpleImputer(strategy="median")

X_train, X_leftover, y_train, y_leftover = train_test_split(X, y, train_size=0.5)
X_cal, X_val, y_cal, y_val = train_test_split(X_leftover, y_leftover, train_size=0.7)

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_cal = imputer.transform(X_cal)


upperReg = QuantileRegressor(
    quantile=0.95,
    alpha=1e-3

)

lowerReg = QuantileRegressor(
    quantile=0.05,
    alpha=1e-3
)

upperReg.fit(X_train, y_train)
lowerReg.fit(X_train, y_train)

print(upperReg.coef_)
print(lowerReg.coef_)
q_low = lowerReg.predict(X_cal)
q_high = upperReg.predict(X_cal)

scores = np.maximum(q_low - y_cal.to_numpy(), y_cal.to_numpy()-q_high)
N = len(scores)
print(N)
quantile = np.ceil((N + 1) * (1 - alpha))/N
q_hat = np.quantile(scores, quantile)

upper_val = upperReg.predict(X_val)
lower_val = lowerReg.predict(X_val)
upper_conformal = upper_val + q_hat
lower_conformal = lower_val - q_hat

y_val_array = y_val.to_numpy()

coverage = np.mean(
    (lower_val <= y_val_array) & (upper_val >= y_val_array)
)
conformalcoverage = np.mean(
    (lower_conformal <= y_val_array) & (upper_conformal >= y_val_array)
)


print(coverage, conformalcoverage, q_hat)


# Sort by the true y value (optional, but makes the plot easier to read)
sort_idx = np.argsort(y_val_array)

y_sorted = y_val_array[sort_idx]

lower_model_sorted = lower_val[sort_idx]
upper_model_sorted = upper_val[sort_idx]

lower_conf_sorted = lower_conformal[sort_idx]
upper_conf_sorted = upper_conformal[sort_idx]

x = np.arange(len(y_sorted))

fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(
    x,
    lower_model_sorted,
    upper_model_sorted,
    color="royalblue",
    alpha=0.25,
    label="Model 5%-95% interval"
)

# Conformal 5%-95% interval
ax.fill_between(
    x,
    lower_conf_sorted,
    upper_conf_sorted,
    color="orange",
    alpha=0.25,
    label="Conformal 5%-95% interval"
)

# Model quantile predictions
ax.scatter(
    x,
    lower_model_sorted,
    color="blue",
    linewidth=1,
    label="Model lower quantile"
)

ax.scatter(
    x,
    upper_model_sorted,
    color="blue",
    linewidth=1,
    label="Model upper quantile"
)

# Conformal bounds
ax.scatter(
    x,
    lower_conf_sorted,
    color="darkorange",
    linestyle="--",
    linewidth=1,
    label="Conformal lower bound"
)

ax.scatter(
    x,
    upper_conf_sorted,
    color="darkorange",
    linestyle="--",
    linewidth=1,
    label="Conformal upper bound"
)

# True y values
ax.scatter(
    x,
    y_sorted,
    color="black",
    s=15,
    alpha=0.7,
    label="True y"
)

ax.set_xlabel("Validation observations (sorted by true y)")
ax.set_ylabel("y")
ax.set_title(
    f"Quantile Regression vs. Conformal Prediction Intervals\n"
    f"Model coverage = {coverage:.3f}, "
    f"Conformal coverage = {conformalcoverage:.3f}, "
    f"alpha = {alpha}"
)

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("graphs/regressionCQRscatter")
plt.show()
