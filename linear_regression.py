import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


model = LinearRegression()

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('all_seasons.csv')
print(len(df))




def linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    # R^2 score
    r2 = model.score(X, y)
    print("R²:", r2)

    n, p = X.shape
    adj_r2_full = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("Adj R² (with x2):", adj_r2_full)


    X_const = sm.add_constant(X)  # add intercept term
    ols_model = sm.OLS(y, X_const).fit()
    print(ols_model.summary())
    
all_attributes = df[["gp", "reb", "ast",
        "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct",
        "player_weight", "player_height"]]

just_height = df[["gp", "reb", "ast",
        "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct",
        "player_height"]]

just_weight = df[["gp", "reb", "ast",
        "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct",
        "player_weight"]]

performace_attributes = df[["gp", "reb", "ast",
        "net_rating", "oreb_pct", "dreb_pct", "usg_pct", "ts_pct", "ast_pct",]]

y = df["pts"]

linear_regression_model(performace_attributes, y)
linear_regression_model(just_weight, y)
linear_regression_model(just_height, y)
linear_regression_model(all_attributes, y)
