import joblib
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# read in model and the 100 datapoint test samples
model = joblib.load("models/final_model.pkl")
df = pd.read_csv("assets/test_samples.csv")

# X and y from the 100 datapoints
X_test = df.drop("cardio", axis = 1)
y_test = df["cardio"]

# predict probabilities and class
probabilities = model.predict_proba(X_test) # probabilities
y_pred = model.predict(X_test) # classification

# print class report
print(classification_report(y_test, y_pred))

# save results cm as .png
cm = confusion_matrix(y_test, y_pred)
fig = ConfusionMatrixDisplay(cm, display_labels = ["No", "Yes"]).plot()
plt.suptitle("Production Model Results")
plt.savefig("assets/production_model_results.png")

# create a dataframe of predicted probabilities
df_probabilities = pd.DataFrame(probabilities, columns  = ["probability class 0", "probability class 1"])

# combine probabilities and cardio column
df_combined = pd.concat([df_probabilities, df["cardio"]], axis = 1)

# sort values by cardio firstly, and secondly on probability class 1
df_combined = df_combined.sort_values(by = ["cardio", "probability class 1"], ascending = False)

# finally, save dataframe as csv file
df_combined.to_csv("assets/prediction.csv")

