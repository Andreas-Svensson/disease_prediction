import pandas as pd

from scripts import feature_target, split, scale

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# read dataset
df = pd.read_csv("assets/cardio_train_cleaned.csv")



# create 2 datasets:

# one with categorical features - df_cat
df_cat = df.drop(["ap_hi", "ap_lo", "height", "weight", "bmi"], axis = 1)
df_cat = pd.get_dummies(df_cat, columns = ["bmi_category", "bp_category", "gender"], prefix = ["bmi_cat", "bp_cat", "sex"]) # dummy encoding features

# one with non-categorical features - df_raw
df_raw = df.drop(["bmi_category", "bp_category", "height", "weight"], axis = 1)
df_raw = pd.get_dummies(df_raw, columns = ["gender"], prefix = ["sex"]) # dummy encoding features



# separate feature and target variables (X and y)
X_cat, y_cat = feature_target(df_cat, target = "cardio") # cat
X_raw, y_raw = feature_target(df_raw, target = "cardio") # raw



# split into train|val|test data based on test_val_size
test_val_size = 0.2 # 0.2 gives train|val|test of 0.8|0.2|0.2
X_cat_train, X_cat_val, X_cat_test, y_cat_train, y_cat_val, y_cat_test = split(X_cat, y_cat, test_val_size) # cat
X_raw_train, X_raw_val, X_raw_test, y_raw_train, y_raw_val, y_raw_test = split(X_raw, y_raw, test_val_size) # raw



# scale X data:

# one standardized
scaler_std = StandardScaler()
X_cat_train_scaled_std, X_cat_val_scaled_std = scale(X_cat_train, X_cat_val, scaler = scaler_std)
X_raw_train_scaled_std, X_raw_val_scaled_std = scale(X_raw_train, X_raw_val, scaler = scaler_std)


# one normalized
scaler_minmax = MinMaxScaler()
X_cat_train_scaled_minmax, X_cat_val_scaled_minmax = scale(X_cat_train, X_cat_val, scaler = scaler_minmax)
X_raw_train_scaled_minmax, X_raw_val_scaled_minmax = scale(X_raw_train, X_raw_val, scaler = scaler_minmax)
