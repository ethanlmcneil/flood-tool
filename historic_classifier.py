"""
Author: team Thames

The historic classifier methods were adopted in tool.py. This is the source code used during development.
"""

import pandas as pd
import numpy as np
import geo

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, make_scorer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# The scoring matrices based on the scoring method listed on the project description page.
# The order is TN, FP, FN, TP, and the weighting is the mean value of each section area.
SCORE_MATRIX = {
    "balanced": (750, 105, -3210, 10160),                   # TN section being [0:5, 0:4]
    "recall_precision_focused": (74, 47, -397.5, 1486.33),  # Each section is the 3 * 3 square in each corner
    "ones": (1, -1, -1, 1)                                  # Weight all cases evenly
}

PREDICTION_TYPES = ["f1",
                    "recall",
                    "precision",
                    "accuracy",
                    "score_maxtrix_weighted_balanced",
                    "score_maxtrix_weighted_recall_precision_focused"]

def df_to_lat_long_zero_origin(df):
    """
    Convert the easting and northing coordinates to standardised longtitudes and latitudes.

    Parameters
    -----------

    df: pandas.Dataframe
        The input dataframe to be converted.
    
    Returns
    -----------
    
    df: pandas.Dataframe
        The converted outcome.

    """
    # Check if required columns are present
    if 'easting' not in df.columns or 'northing' not in df.columns:
        raise ValueError("DataFrame must contain 'easting' and 'northing' columns")

    # Convert easting and northing to latitude and longitude
    df['latitude'], df['longitude'] = geo.get_gps_lat_long_from_easting_northing(df['easting'], df['northing'])

    # Ensure no NaN values in latitude and longitude
    if df['latitude'].isnull().any() or df['longitude'].isnull().any():
        raise ValueError("Conversion to latitude and longitude resulted in NaN values")

    # Shift origin to zero
    df['latitude'] = df['latitude'] - df['latitude'].min()
    df['longitude'] = df['longitude'] - df['longitude'].min() 

    # Scale the data
    df['latitude_scaled'] = df['latitude'] / df['latitude'].max()
    df['longitude_scaled'] = df['longitude'] / df['longitude'].max() 

    # Drop intermediate latitude and longitude columns
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)   

    return df

def preprocess(data):
    """
    Preprocess the input dataframe and return splitted train/test dataset.
    Preprocessing includes dropping the features with too many missing data points, 
    standardisation and encoding.

    Parameters
    -----------

    data: pandas.Dataframe
        The input dataframe to be preprocessed.
    
    Returns
    -----------
    
    X_train, X_test, y_train, y_test: pandas.Dataframe
        The converted outcome.

    """
    data = data.drop_duplicates()
    y = data['historicallyFlooded']
    data = df_to_lat_long_zero_origin(data).drop(
        ['easting', 'northing', 'postcode', 'historicallyFlooded', 'riskLabel', 'medianPrice'], errors='ignore', axis=1)

    # Drop columns with >30% missing data points
    missing = data.isnull().sum() / len(data)
    features_to_drop = missing[missing > 0.3].index
    data = data.drop(columns=features_to_drop)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42) 

    X_train = preprocess_X(X_train)
    X_test = preprocess_X(X_test)

    return X_train, X_test, y_train, y_test


def preprocess_X(X):
    """
    Preprocess the dataset with target labels excluded.

    Parameters
    -----------

    X: pandas.Dataframe
        The input dataframe to be preprocessed.
    
    Returns
    -----------
    
    X: pandas.Dataframe
        The preprocessed dataset.

    """
    # Impute missing numeric features
    num_features = X.select_dtypes(include=['float64', 'int64']).columns
    cat_features = X.select_dtypes(include=['object']).columns
    num_imputer = SimpleImputer(strategy='mean')
    X[num_features] = num_imputer.fit_transform(X[num_features])

    # Standardise numeric features
    ss = StandardScaler()
    X_scaled = ss.fit_transform(X[num_features])
    X_df_scaled = pd.DataFrame(X_scaled, columns=ss.get_feature_names_out(num_features), index=X.index)

    # Encode categorical features
    ohe = OneHotEncoder(sparse_output=False)
    encoded_cat = ohe.fit_transform(X[cat_features])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(cat_features), index=X.index)

    # Final dataset
    X = pd.concat([X_df_scaled, encoded_cat_df], axis=1)

    return X


# A configurable scoring function based on FN, FP, TN, TP
def scoring_core(y_true, y_pred, mode):
    """
    The scoring function for the best model selection in RandomizedSearchCV().

    Parameters
    -----------

    y_true, y_pred: pandas.Dataframe
        The input dataframe to be preprocessed.

    mode: String
        The user selected scoring method.
    
    Returns
    -----------
    
    score: float
        The final score of the preditions.

    """
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Define weights for each component
    avg_tn, avg_fp, avg_fn, avg_tp = SCORE_MATRIX[mode]

    score = ((avg_tp * tp) + (avg_fp * fp) + (avg_fn * fn) + (avg_tn * tn)) / len(y_true)
    return score

# Use a separated funtion to allow manual parameter changes
def custom_scorer(mode):
    """
    Encapsulate the make_scorer function to allow user defined scoring method.

    Parameters
    -----------

    mode: String
        The user selected scoring method.
    
    Returns
    -----------
    
    sklearn.metrics.make_scorer
        The scorer object for RandomizedSearchCV(). 

    """
    return make_scorer(scoring_core, mode=mode)


def train_model(X_train, X_test, y_train, y_test, mode, undersampling_ratios=range(55, 100, 5)):
    """
    The main function used for model training.

    Parameters
    -----------

    X_train, X_test, y_train, y_test: pandas.Dataframe
        The input data for training.
    
    mode: String
        The user selected scoring method for RandomizedSearchCV().
        Supported ones are included in SCORE_MATRIX.

    undersampling_ratios: range(integer)
        This model needs to be undersampled for improved 
        performance in recall and precision. Each instance is the percentage
        (in integer) of the negative data points to be removed before training
        in each iteration.
    
    Returns
    -----------
    
    best_models: List(Object)
        List of best models selected in each iteration.
        Can be used for future predictions. 

    num_results: pandas.Dataframe
        Lists of the numeric outcomes of each iteration.
        Can be used for plotting curves.

    """
    np.random.seed(42)

    param_distrib = [
        {'classifier__max_depth': randint(low=1, high=50),
        'classifier__n_estimators': randint(low=100, high=500),
        'classifier__min_samples_split': randint(low=2, high=10),
        'classifier__max_features': randint(low=1, high=8)}]

    model_pipe = Pipeline([('classifier',RandomForestClassifier())])

    negative_indices = y_train[y_train == 0].index

    scoring = custom_scorer(mode) if mode in SCORE_MATRIX else mode

    num_result_list = []
    best_models = []
    for iteration, ur in enumerate(undersampling_ratios, start=1):    
        print(f"Current under-sampling ratio: {ur}% ({iteration} / {len(undersampling_ratios)})")

        # Drop most of the negative data points
        num_to_drop = int(len(negative_indices) * ur / 100)  
        rows_to_drop = np.random.choice(negative_indices, num_to_drop, replace=False)

        X_train_reduced = X_train.drop(rows_to_drop)
        y_train_reduced = y_train.drop(rows_to_drop)


        rnd_search = RandomizedSearchCV(model_pipe, param_distributions=param_distrib, cv=5, n_iter=10,
                            scoring=scoring, n_jobs=-1, random_state=1)

        rnd_search.fit(X_train_reduced, y_train_reduced)
        rnd_model = rnd_search.best_estimator_


        y_test_pred = rnd_model.predict(X_test)

        best_models.append(rnd_model)
        num_result_list.append({
            'drop_percentage': ur,
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'target_score': scoring_core(y_test, y_test_pred, mode) if mode in SCORE_MATRIX else None,
        })

    num_results = pd.DataFrame(num_result_list)

    return best_models, num_results


def get_predictions(X, prediction_type, best_models, num_results):
    """
    Use the trained models to predict the historic flood records. 
    The final model will be selected by user preference. 
    Supported ones can be found in PREDICTION_TYPES.
    Parameters
    -----------

    X: pandas.Dataframe
        The input data for prediction.
    
    prediction_type: String
        User can choose one evaluation method for classification in PREDICTION_TYPES.

    best_models: List(Object)
        List of best models selected in each iteration.
        Can be used for future predictions. 

    num_results: pandas.Dataframe
        Lists of the numeric outcomes of each iteration.
        Can be used for plotting curves.

    Returns
    -----------
    
    y_pred: list
    The final prediction result

    """
    if prediction_type not in PREDICTION_TYPES:
        raise ValueError("Unexpected prediction type")

    max_index = 0
    # The method focusing on the final overall score
    if prediction_type == "score_maxtrix_weighted_balanced":
        num_results['performance'] = (10160 + 3210) * num_results['recall'] + (10160 - 105) * num_results['precision']
        max_index = num_results['performance'].idxmax()

    # The method focusing on recall and precision
    if prediction_type == "score_maxtrix_weighted_recall_precision_focused":
        num_results['performance'] = (1486.33 + 397.5) * num_results['recall'] + (1486.33 - 47) * num_results['precision']
        max_index = num_results['performance'].idxmax()

    # The classic scoring methods
    if prediction_type in ["f1", "recall", "precision", "accuracy"]:
        max_index = num_results[prediction_type].idxmax()

    best_model = best_models[max_index]
    y_pred = best_model.predict(X)
    return y_pred

def plot_train_model_results(results_df):
    """
    Plot the results with different percentages of negative data points removed.
    Parameters
    -----------

    results_df: pandas.Dataframe
        The scoring data generated during the main training loop.

    """
    drop_percentage_range = results_df['drop_percentage']
    precisions = results_df['precision']
    recalls = results_df['recall']
    accuracies = results_df['accuracy']
    f1_s = results_df['precision'] * results_df['recall'] / (results_df['precision'] + results_df['recall'])

    # Plot the precision, recall, and accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(drop_percentage_range, precisions, label='Precision')
    plt.plot(drop_percentage_range, recalls, label='Recall')
    plt.plot(drop_percentage_range, accuracies, label='Accuracy')
    plt.plot(drop_percentage_range, f1_s, label='F1-Score')
    plt.xlabel('Percentage of Negative Data Points Removed')
    plt.ylabel('Score')
    plt.title('Precision, Recall & Accuracy vs. Percentage of Negative Data Points Removed')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    data = pd.read_csv("flood_tool/resources/postcodes_labelled.csv")
    unlabelled_data = pd.read_csv("flood_tool/example_data/postcodes_unlabelled.csv")


    X_train, X_test, y_train, y_test = preprocess(data)
    best_models, num_results = train_model(X_train, X_test, y_train, y_test, 'f1')

    # plot_train_model_results(num_results) # Uncomment this line to plot

    X_preprocessed = preprocess_X(df_to_lat_long_zero_origin(unlabelled_data).drop(
            ['easting', 'northing', 'postcode', 'historicallyFlooded', 'nearestWatercourse', 'riskLabel', 'medianPrice'], errors='ignore', axis=1))
    unlabelled_pred = get_predictions(X_preprocessed, 'f1', best_models, num_results)

    print(unlabelled_pred.shape)
    unlabelled_pred.describe()

