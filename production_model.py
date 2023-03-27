import joblib
import pandas as pd

# read in model and the 100 datapoint test samples
model = joblib.load("models/final_model.pkl")
df = pd.read_csv("assets/test_samples.csv")

probabilities = model.predict_proba(df.drop("cardio", axis = 1))

# create a dataframe of predicted probabilities
df_probabilities = pd.DataFrame(probabilities, columns  = ["probability class 0", "probability class 1"])

# combine probabilities and cardio column
df_combined = pd.concat([df_probabilities, df["cardio"]], axis = 1)

# sort values by cardio firstly, and secondly on probability class 1
df_combined = df_combined.sort_values(by = ["cardio", "probability class 1"], ascending = False)

# finally, save dataframe as csv file
df_combined.to_csv("assets/prediction.csv")

