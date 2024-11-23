



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import adjusted_rand_score


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score



from sklearn.cluster import KMeans, DBSCAN

from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
from geo import get_gps_lat_long_from_easting_northing, lat_long_to_xyz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy



def custom_score(y_true, y_pred):


    # Generate the confusion matrix

    scoring_matrix =  np.array([
    [100,   80,   60,   40,    0,  -40,  -60],
    [ 60,  100,   80,   60,   45,    0,  -20],
    [ 30,   90,  150,  120,   90,   60,   30],
    [  0,   60,  120,  150,  120,   90,   60],
    [-30,   30,   60,  120,  150,  120,   90],
    [-600,    0,   30,   90,  900, 1500, 1200],
    [-1800, -600, -300,    0, 1200, 2000, 3000]
])

    conf_matrix = confusion_matrix(y_true, y_pred)
    
    score = 0
    for i in range(conf_matrix.shape[0]):  # Rows
        for j in range(conf_matrix.shape[1]):  # Columns
            score += conf_matrix[i, j] * scoring_matrix[i, j]
    
    return score


custom_scorer = make_scorer(custom_score, greater_is_better=True)






from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class optimize_model:
   

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.baseline_model = None

    def optimize(self, model, param_grid, n_iter=10, refinements=3, factor=0.5):
      
        self.model = model

        for refinement in range(refinements):
            print(f"\n=== Refinement Round {refinement + 1}/{refinements} ===")

            # Perform Randomized Search
            random_search = RandomizedSearchCV(
                self.model, 
                param_distributions=param_grid, 
                n_iter=n_iter, 
                cv=3, 
                scoring=custom_scorer, 
                return_train_score=True, 
                random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            # Store the best parameters and score
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            self.best_model = random_search.best_estimator_

            # Display results
            print("Best Parameters:", self.best_params)
            print("Best Custom Training Score:", self.best_score)

            # Test data performance of the best model
            y_pred = self.best_model.predict(self.X_test)
            best_score_on_test = custom_score(self.y_test, y_pred)
            print("Best Model Score on Test Data:", best_score_on_test)

            # Refine the parameter grid
            param_grid = self.refine_param_grid(self.best_params, param_grid, factor=factor)

        # Baseline model (without hyperparameter tuning)
        self.baseline_model = self.model.__class__()  # Initialize a fresh instance of the same model
        self.baseline_model.fit(self.X_train, self.y_train)
        y_pred_baseline = self.baseline_model.predict(self.X_test)
        baseline_score = custom_score(self.y_test, y_pred_baseline)
        print("\nBaseline Model Score on Test Data:", baseline_score)

        return self.best_params

    def refine_param_grid(self, best_params, param_grid, factor=0.5):
        """
        Refine the parameter grid around the best parameters found in the last run.
        
        Args:
            best_params: The best parameters from the previous run.
            param_grid: The current parameter grid.
            factor: The percentage by which to refine the search range (default is 0.5).
        
        Returns:
            dict: The refined parameter grid.
        """
        refined_grid = {}
        for param, values in param_grid.items():
            if isinstance(values, list) and param in best_params:
                best_value = best_params[param]
                if isinstance(best_value, int):
                    # Narrow the range for integer parameters
                    low = max(min(values), int(best_value - factor * (max(values) - min(values))))
                    high = min(max(values), int(best_value + factor * (max(values) - min(values))))
                    refined_grid[param] = list(range(low, high + 1))
                elif isinstance(best_value, float):
                    # Narrow the range for float parameters
                    low = max(min(values), best_value - factor * (max(values) - min(values)))
                    high = min(max(values), best_value + factor * (max(values) - min(values)))
                    refined_grid[param] = np.linspace(low, high, num=5).tolist()
                else:
                    # For categorical values, keep the original set
                    refined_grid[param] = values
            else:
                # If no refinement is possible, retain the original parameter values
                refined_grid[param] = values

        return refined_grid










