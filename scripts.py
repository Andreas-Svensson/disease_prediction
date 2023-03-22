from sklearn.model_selection import train_test_split, GridSearchCV


def feature_target(df, target):
    """Create X and y variable based on dataframe and target feature (column name of dataframe)"""
    X = df.drop(target, axis = 1)
    y = df[target]

    return X, y


def split(X, y, test_val_size, val = True):
    """Split X and y into test, val, train based on test_val_size"""
    N = test_val_size
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = N, random_state = 42)

    if val: # also split into validation data if val = True
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = N / (1 - N), random_state = 42) # N/(1-N) gives approx N of the original dataset's size (i.e. same as test size for val)
        return X_train, X_val, X_test, y_train, y_val, y_test # return data with val
    return X_trainval, X_test, y_trainval, y_test # return data without val


def scale(X_train, X_val, X_test, scaler):
    """Scale data based on input scaler, returns scaled data"""
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def evaluate_grid_search(grid_search, param_grid) -> None:
    """Print out gridsearch's best parameters out of possible available parameters in a neatly formatted manner"""
    for i in grid_search.best_params_:
        print(f"{i}:")
        print(f"'{grid_search.best_params_[i]}' chosen out of possible {param_grid[i]}")
        print()


def times_model_ran(param_grid, cv = 5) -> None:
    """Calculate how many times a model is fitted using GridSearchCV"""
    times = 1

    for i in param_grid: # calculate amount of settings in param grid
        times *= len(param_grid[i])

    times *= cv # multiply by amount cross-validations

    return times


def perform_grid_search(X_val, y_val, param_grid, model, cv = 5, n_jobs = -1, scoring = "recall", evaluate = False):
    """Instantiate GridSearchCV based on parameters, fit based on val data, evaluate = True prints out best parameters found"""
    if evaluate:
        times = times_model_ran(param_grid, cv)
        print(f"Model is being fitted {times} times...")

    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        cv = cv,
        n_jobs = n_jobs,
        scoring = scoring
    )

    grid_search.fit(X_val, y_val)

    if evaluate == True: # print evaluation results
        evaluate_grid_search(grid_search, param_grid)

    return grid_search


