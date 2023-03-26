# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils._testing import ignore_warnings # used to suppress warnings from grid search fit



def calculate_outliers(data):
    """Use Tukey's fence to calculate outliers, return lower and upper boundaries"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    outlier_lower = Q1 - 1.5 * IQR
    outlier_upper = Q3 + 1.5 * IQR

    return outlier_lower, outlier_upper


def feature_target(df, target):
    """Create X and y variable based on dataframe and target feature (column name of dataframe)"""
    X = df.drop(target, axis = 1)
    y = df[target]

    return X, y


def violin_outliers_by_gender(outliers: dict, df, feature) -> dict:
    """Input dictionary of outliers as key (feature): value (tuple: lower, upper)"""
    
    # calculate and add outliers to dict as key (gender) and value (tuple: lower outlier, upper outlier)
    # outliers = {"Women": calculate_outliers(df[df["gender"] == 1][feature]), "Men": calculate_outliers(df[df["gender"] == 2][feature])}

    # set up subplots
    fig, axes = plt.subplots(1, len(outliers), figsize = (8 * len(outliers), 4))

    # loop over outliers and axes
    for i, (gender, ax) in enumerate(zip(outliers, axes.flatten())):

        # plot feature data
        sns.violinplot(data = df[df["gender"] == i+1], y = feature, ax = ax)
        ax.set_title(gender)

        # plot horizontal lines for outliers
        ax.axhline(y = outliers[gender][1], label = f"Upper bound outlier = {outliers[gender][1]:.1f}", color = "r", linestyle = "--", alpha = 0.7)
        ax.axhline(y = outliers[gender][0], label = f"Lower bound outlier = {outliers[gender][0]:.1f}", color = "r", linestyle = "--", alpha = 0.7)
        ax.legend(loc = "upper right")

    plt.suptitle(f"{feature.capitalize()} Distribution and Outliers by Gender")
    # return dictionary of key (gender) and value (tuple: lower outlier, upper outlier)
    # return outliers


def split(X, y, test_val_size, val = True):
    """Split X and y into test, val, train based on test_val_size"""
    N = test_val_size
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = N, random_state = 42)

    if val: # also split into validation data if val = True
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = N / (1 - N), random_state = 42) # N/(1-N) gives approx N of the original dataset's size (i.e. same as test size for val)
        return X_train, X_val, X_test, y_train, y_val, y_test # return data with val
    return X_trainval, X_test, y_trainval, y_test # return data without val


def scale(X_train, X_val, scaler):
    """Scale data based on input scaler, returns scaled data"""
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled


def times_model_ran(param_grid, cv = 5) -> None:
    """Calculate how many times a model is fitted using GridSearchCV"""
    times = 1

    for i in param_grid: # calculate amount of settings in param grid
        times *= len(param_grid[i])

    times *= cv # multiply by amount cross-validations

    return times


def display_gridsearch_results(grid_search, param_grid):
    """Print best parameters found next to list of possible parameters gridsearch had available"""

    # pick out best parameters
    best_params = grid_search.best_params_

    print(f"Grid search results")
    print(f"\n-------------------\n")

    # iterate over parameter keys
    for i, param in enumerate(best_params):

        # print out best parameter out of possible parameters in param_grid
        print(f"'{param}' best param: {best_params[param]}")
        print(f" out of: {param_grid[param]}\n")

    print(f"-------------------\n")
    
    # finally print the score matching above parameters
    print(f"Best score: {grid_search.best_score_}")



def custom_scorer(y_true, y_pred):
    """Calculate GridSearchCV score based on weights, used as default scorer in perform_grid_search """
    recall_weight = 2  # weight towards recall
    precision_weight = 1
    accuracy_weight = 1
    
    # compute precision, recall, and accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    
    # combine scores multiplied by their respectice weight, divide by sum of all weights
    score = (recall_weight * recall + precision_weight * precision + accuracy_weight * accuracy) / (recall_weight + precision_weight + accuracy_weight)
    
    return score


@ignore_warnings() # suppress warnings from grid search fit
def perform_grid_search(X, y, param_grid, model, cv = 5, n_jobs = -1, scoring = make_scorer(custom_scorer), display_results = True):
    """Instantiate GridSearchCV based on model and parameters, fit based on X, y data"""

    # instantiate grid_search
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = cv,
        n_jobs = n_jobs,
        scoring = scoring,
    )

    grid_search.fit(X, y) # fit to data

    if display_results: # print out best parameters found
        display_gridsearch_results(grid_search, param_grid)

    return grid_search


def tune_param_grid(grid_search, param_grid: dict) -> dict:
    """Tune param_grid values based on GridSearchCV results, returns updated param_grid"""
    # store old values
    old = dict(param_grid) 

    # loop over results of best grid search params
    for key, value in grid_search.best_params_.items():

        # if numeric:
        if type(value) == int or type(value) == float:

            # if lowest value was best:
            if value == param_grid[key][0]:
                # calculate new upper and lower values for that parameter
                low = value / 2 # new lower value is half of current

                diff_up = param_grid[key][1] - param_grid[key][0] # calculate difference between old middle and low values
                high = value + diff_up / 2 # new higher value is old value increased by half the distance to old high
            
            # if middle value was best:
            if value == param_grid[key][1]:
                # calculate new upper and lower values for that parameter
                diff_down = param_grid[key][1] - param_grid[key][0] # calculate difference between old middle and low values
                low = value - diff_down / 2 # new lower value is old value decreased by half the distance to old low

                diff_up = param_grid[key][2] - param_grid[key][1] # calculate difference between old high and middle values
                high = value + diff_up / 2 # new higher value is old value increased by half the distance to old high
            
            # if highest value was best:
            if value == param_grid[key][2]:
                # calculate new upper and lower values for that parameter
                diff_down = param_grid[key][2] - param_grid[key][1] # calculate difference between old high and middle values
                low = value - diff_down / 2 # new lower value is old value decreased by half the distance to old low

                high = value * 2 # new higher value is twice of current

            # update parameter with new calculated low, middle, and high values to use in next grid search
            param_grid[key] = [round(low, 3), round(value, 3), round(high, 3)] # round to not get into crazy decimals
    
    # display changes made
    for key, values in param_grid.items():

        # if numerical:
        if type(values[0]) == int or type(values[0]) == float:

            # print old parameter values compared to new
            print(f"'{key}' parameter values: (best {grid_search.best_params_[key]})")
            print(f"old: {old[key]}")
            print(f"new: {values}")
            print()

    return param_grid


def evaluate_predictions_plot(ys: list, y_preds: list, suptitle: str, subplot_titles: list = ["Categorical Standardized", "Categorical Normalized", "Non-Categorical Standardized", "Non-Categorical Normalized"]):
    """2x2 confusion matrix plot, returns figure"""

    fig, axes = plt.subplots(2, 2, figsize = [10, 6])
    
    # plot confusion matrix
    for ax, y, y_pred, title in zip(axes.flatten(), ys, y_preds, subplot_titles):
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay(cm, display_labels = ["No", "Yes"]).plot(ax = ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.suptitle(suptitle, x = 0.55, y = 1.05, horizontalalignment = "center", fontsize = 16) # for whatever reason, 0.55 is the center of the figure

    return fig

