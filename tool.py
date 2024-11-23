'''Example module in template package.'''

import os

from collections.abc import Sequence
from typing import List

from matplotlib import _preprocess_data
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from .geo import *  # noqa: F401, F403

__all__ = [
    'Tool',
    '_data_dir',
    '_example_dir',
    'flood_class_from_postcode_methods',
    'flood_class_from_location_methods',
    'house_price_methods',
    'local_authority_methods',
    'historic_flooding_methods',
]

_data_dir = os.path.join(os.path.dirname(__file__), 'resources')
_example_dir = os.path.join(os.path.dirname(__file__), 'example_data')


flood_class_from_postcode_methods = {
    'Random_Forest_flood': 'Random Forest Classifier'
}
flood_class_from_location_methods = {
    'Random_Forest_flood': 'Random Forest Classifier'
}
historic_flooding_methods = {
    'Random_Forest_hist': 'Random Forest Classifier'
}
house_price_methods = {
    'Random_Forest_house': 'Random Forest Regressor'
}
local_authority_methods = {
    'Random_Forest_local': 'Random Forest Classifier'
}

IMPUTATION_CONSTANTS = {
    'soilType': 'Unsurveyed/Urban',
    'elevation': 60.0,
    'nearestWatercourse': '',
    'distanceToWatercourse': 80,
    'localAuthority': np.nan
}
IMPUTATION_STRATEGIES = {
    'soilType': 'most_frequent',
    'elevation': 'mean',
    'nearestWatercourse': 'most_frequent',
    'distanceToWatercourse': 'mean'
}
best_params = {
    'n_estimators': 83,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_depth': 20,
    'class_weight': 'balanced_subsample',
    'bootstrap': False
    }
Parameters_house ={
 'random_state': 42,
 'n_estimators': 140,
 'min_samples_split': 3,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 42,
 'bootstrap': True,
 }




class Tool(object):
    '''Class to interact with a postcode database file.'''

    def __init__(self, labelled_unit_data: str = '',
                 unlabelled_unit_data: str = '',
                 sector_data: str = '', district_data: str = '',
                 additional_data: dict = {}):

        '''
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additional .csv files containing addtional
            information on households.
        '''

        # Set defaults if no inputs provided
        if labelled_unit_data == '':
            labelled_unit_data = os.path.join(_data_dir,
                                              'postcodes_labelled.csv')

        if unlabelled_unit_data == '':
            unlabelled_unit_data = os.path.join(_example_dir,
                                                'postcodes_unlabelled.csv')

        if sector_data == '':
            sector_data = os.path.join(_data_dir,
                                       'sector_data.csv')
        if district_data == '':
            district_data = os.path.join(_data_dir,
                                         'district_data.csv')
        

        self._postcodedb = pd.read_csv(labelled_unit_data)
        self._unlabelled_data = pd.read_csv(unlabelled_unit_data)
        self._sector_data = pd.read_csv(sector_data)
        self._district_data = pd.read_csv(district_data)
        self._additional_data = {}
        self._top_categories = {}
        self._models = {}
        research_unit_data = os.path.join(_data_dir,
                                              'soil_details.csv')

        self._research_data = pd.read_csv(research_unit_data,encoding='ISO-8859-1')
        for key, filepath in additional_data.items():
            self._additional_data[key] = pd.read_csv(filepath)


    def split_postcode(self, postcode: str) -> str:
        '''
        Split a full postcode into the postcode unit.

        Parameters
        ----------

        postcode : str
            Full postcode.

        Returns
        -------
        str
            Postcode unit.
        '''

        parts = postcode.split(' ')

        outcode = parts[0] 
        incode_first_digit = parts[1][0]

        return f'{outcode} {incode_first_digit}'



        return postcode.split(' ')[0]

    def get_data(self, data_name: str):
        '''
        Retrieve and return data based on the data name.

        Parameters
        ----------
        data_name : str
            Name of the data set to retrieve (e.g., 'postcodedb', 'unlabelled', 'sector', 'district', 'additional').

        Returns
        -------
        DataFrame
            The requested data as a pandas DataFrame.
        '''
        if data_name == 'postcodedb':
            return self._postcodedb
        elif data_name == 'unlabelled':
            return self._unlabelled_data
        elif data_name == 'sector':
            return self._sector_data
        elif data_name == 'district':
            return self._district_data
        elif data_name == 'additional':
            return self._additional_data
        else:
            raise ValueError("Invalid data name provided.")
        
    def preprocess_hist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess historical property data for modeling.

        This method performs the following steps:
        - Drops unnecessary columns.
        - Imputes missing values for categorical and numerical features.
        - Encodes categorical variables using one-hot encoding.
        - Scales numerical features to a [0, 1] range.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing historical property data.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame ready for modeling.
    
        Raises
        ------
        KeyError
            If any of the columns to drop are not found in the DataFrame.
        """

        df = df.copy()
        features_drop = ['medianPrice', 'riskLabel','nearestWatercourse','localAuthority']
        for variable in features_drop:
            if variable in df.columns:
                df = df.drop(variable, axis=1)

        num_data_cols = df.select_dtypes(include=[np.number]).columns
        cat_data_cols = df.select_dtypes(include=[object]).columns
        cat_data_cols = cat_data_cols.difference(['postcode','historicallyFlooded'])
        num_data_cols = num_data_cols.difference(['postcode','historicallyFlooded'])
        for col in cat_data_cols:
            imputed_categoric = self.impute_missing_values(df[[col]], method='most_frequent')
            df[col] = imputed_categoric[col]

        for col in num_data_cols:
            imputed_numeric = self.impute_missing_values(df[[col]], method='mean')  
            df[col] = imputed_numeric[col] 

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df[cat_data_cols]) 
        encoded_features = encoder.transform(df[cat_data_cols])
        encoded_feature_names = encoder.get_feature_names_out(cat_data_cols)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
        df = df.drop(columns=cat_data_cols)
        df = pd.concat([df, encoded_df], axis=1) 

        scaler = MinMaxScaler()
        df[num_data_cols] = scaler.fit_transform(df[num_data_cols])
        return df

        
    def preprocess_data_house(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to preprocess data, handle missing values, encode categorical variables,
        and prepare data for modeling.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to preprocess.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame.
        '''  
        df = df.copy()
    
    # Define columns to drop
        features_drop = [
        'nearestWatercourse',                                                            
        'riskLabel',                                                                      
        'historicallyFlooded',                                                             
        'Description',                                                                    
        'PSD max(mm)',                                                                   
        ' **Notes**                                                                 ',
        'Bulk Density Max (g/cm³)', 
        'Bulk Density Min (g/cm³)',
        'Porosity Mean(%)',
        'Permeability mean(m²)',
        'Permeability category',
        'localAuthority'
        ]
    
        if 'soilType' not in df.columns or 'soilType' not in self._research_data.columns:
            raise KeyError("'soilType' column must exist in both DataFrames for merging.")
    
        df_merged = pd.merge(df, self._research_data, on='soilType', how='left')
        df_merged.drop(columns=[col for col in features_drop if col in df_merged.columns], inplace=True)
    
        num_data_cols = df_merged.select_dtypes(include=[np.number]).columns
        cat_data_cols = df_merged.select_dtypes(include=[object]).columns
    
    # Exclude 'postcode' and 'medianPrice' from feature columns
        cat_data_cols = cat_data_cols.difference(['postcode','medianPrice'])
        num_data_cols = num_data_cols.difference(['postcode','medianPrice'])
        for col in cat_data_cols:
            imputed_categoric = self.impute_missing_values(df_merged[[col]], method='most_frequent')
            df_merged[col] = imputed_categoric[col]

        for col in num_data_cols:
            imputed_numeric = self.impute_missing_values(df_merged[[col]], method='mean')  
            df_merged[col] = imputed_numeric[col] 
    
    # Encode categorical variables
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df_merged[cat_data_cols]) 
        encoded_features = encoder.transform(df_merged[cat_data_cols])
        encoded_feature_names = encoder.get_feature_names_out(cat_data_cols)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_merged.index)

    # Drop original categorical columns and concatenate encoded ones
        df_merged.drop(columns=cat_data_cols, inplace=True)
        df_merged = pd.concat([df_merged, encoded_df], axis=1) 

        scaler = MinMaxScaler()
        df_merged[num_data_cols] = scaler.fit_transform(df_merged[num_data_cols]) 
    
        return df_merged
            
    def preprocesss_data_flood(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess flood-related property data for modeling.

        This method performs the following steps:
        - Drops unnecessary columns.
        - Imputes missing values for categorical and numerical features.
        - Encodes categorical variables using one-hot encoding.
        - Scales numerical features to a [0, 1] range.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing flood-related property data.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame ready for modeling.
    
        Raises
        ------
        KeyError
            If any of the columns to drop are not found in the DataFrame.
        """
        df = df.copy()
        features_drop = ['medianPrice', 'historicallyFlooded','nearestWatercourse','distanceToWatercourse','localAuthority']
        for variable in features_drop:
            if variable in df.columns:
                df = df.drop(variable, axis=1)

        num_data_cols = df.select_dtypes(include=[np.number]).columns
        cat_data_cols = df.select_dtypes(include=[object]).columns
        cat_data_cols = cat_data_cols.difference(['postcode','riskLabel'])
        num_data_cols = num_data_cols.difference(['postcode','riskLabel'])
        for col in cat_data_cols:
            imputed_categoric = self.impute_missing_values(df[[col]], method='most_frequent')
            df[col] = imputed_categoric[col]

        for col in num_data_cols:
            imputed_numeric = self.impute_missing_values(df[[col]], method='mean')  
            df[col] = imputed_numeric[col] 

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df[cat_data_cols]) 
        encoded_features = encoder.transform(df[cat_data_cols])
        encoded_feature_names = encoder.get_feature_names_out(cat_data_cols)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
        df = df.drop(columns=cat_data_cols)
        df = pd.concat([df, encoded_df], axis=1) 

        scaler = MinMaxScaler()
        df[num_data_cols] = scaler.fit_transform(df[num_data_cols])
        return df

    def preprocess_data_local(self, df: pd.DataFrame) -> pd.DataFrame:
        """
         Preprocess local property data for modeling.

         This method performs the following steps:
         - Drops unnecessary columns.
         - Imputes missing values based on predefined strategies.
         - Handles unspecified columns with default imputation methods.
         - Reduces categorical variables by grouping less frequent categories into 'Other'.
         - Encodes categorical variables using one-hot encoding.
         - Scales numerical features using standard scaling.
         - Retains 'postcode' and 'localAuthority' columns if present.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame containing local property data.

        Returns
        -------
        pandas.DataFrame
            The preprocessed DataFrame ready for modeling.

        Raises
        ------
        KeyError
            If any of the columns to drop are not found in the DataFrame.
        """

        target_variables = ['riskLabel', 'medianPrice', 'historicallyFlooded']
        df = df.copy()
        imputed_dfs = []
        methods = set(IMPUTATION_STRATEGIES.values())
        if 'postcode' in df.columns:
            postcode_col = df[['postcode']]
            local_auth = df[['localAuthority']]
            df = df.drop(columns=['postcode','localAuthority'])
        else:
            postcode_col = None

        for method in methods:
            columns = [col for col in df.columns if IMPUTATION_STRATEGIES.get(col) == method]
            if columns:
                df_subset = df[columns]
                imputed_df = self.impute_missing_values(df_subset, method=method)
                imputed_dfs.append(imputed_df)

        unspecified_columns = [col for col in df.columns if col not in IMPUTATION_STRATEGIES]
        if unspecified_columns:
            df_subset = df[unspecified_columns]
            numeric_cols = df_subset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            non_numeric_cols = df_subset.select_dtypes(include=['object']).columns.tolist()
            if numeric_cols:
                imputed_numeric = self.impute_missing_values(df_subset[numeric_cols], method='mean')
                imputed_dfs.append(imputed_numeric)

            if non_numeric_cols:
                imputed_non_numeric = self.impute_missing_values(df_subset[non_numeric_cols], method='most_frequent')
                imputed_dfs.append(imputed_non_numeric)

        if imputed_dfs:
            df_imputed = pd.concat(imputed_dfs, axis=1)
        else:
            df_imputed = pd.DataFrame()

        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                top_categories = df_imputed[col].value_counts().nlargest(10).index.tolist()
                df_imputed[col] = df_imputed[col].apply(
                    lambda x: x if x in top_categories else 'Other'
                    )


                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoder.fit(df_imputed[categorical_cols])
                encoded_features = encoder.transform(df_imputed[categorical_cols])
                encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(
                    encoded_features, columns=encoded_feature_names, index=df_imputed.index
                    )

            df_imputed.drop(columns=categorical_cols, inplace=True)
            df_imputed = pd.concat([df_imputed, encoded_df], axis=1)

            numerical_cols = df_imputed.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col not in target_variables]
            if 'postcode' in numerical_cols:
                numerical_cols.remove('postcode')
            if numerical_cols:
                scaler = StandardScaler()
                df_imputed[numerical_cols] = scaler.fit_transform(df_imputed[numerical_cols])
        if postcode_col is not None:
            df_imputed = pd.concat([df_imputed, postcode_col], axis=1)
            df_imputed = pd.concat([df_imputed, local_auth], axis=1)        
        return df_imputed
    
    def fit(self, models: List = [], update_labels: str = '',
            update_hyperparameters: bool = False, **kwargs):
        '''Fit/train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to fit/train
        update_labels : str, optional
            Filename of a .csv file containing an updated
            labelled set of samples
            in the same format as the original labelled set.

            If not provided, the data set provided at
            initialisation is used.
        update_hyperparameters : bool, optional
            If True, models may tune their hyperparameters, where
            possible. If False, models will use their default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.fit(fcp_methods[0])  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        '''

        if update_labels:
            self._postcodedb = pd.read_csv(update_labels)
            self._preprocess_data()

        X = self._postcodedb.drop(['riskLabel', 'postcode','medianPrice','historicallyFlooded'], axis=1)
        y_risk = self._postcodedb['riskLabel'] 
        y_price = self._postcodedb['medianPrice']
        y_flooded = self._postcodedb['historicallyFlooded']   
        for model in models:
            if model in flood_class_from_postcode_methods or model in flood_class_from_location_methods:
                X_flood = self.preprocesss_data_flood(self._postcodedb)
                y_flooded = X_flood['riskLabel']
                X = X_flood.drop(['riskLabel','postcode'], axis=1)
                if model == 'Random_Forest_flood':
                    if update_hyperparameters:
                         param_gridRF = {
                            'n_estimators': [10,50,100],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'class_weight': ['balanced', 'balanced_subsample']
 
                         }
                         reg = GridSearchCV(RandomForestClassifier(), param_gridRF, cv=5)
                         reg.fit(X, y_flooded)
                         self._models[model] = reg.best_estimator_
                    else:
                         reg = RandomForestClassifier(**best_params)
                         reg.fit(X, y_flooded)
                         self._models[model] = reg
            if model in house_price_methods:
                 X_price = self.preprocess_data_house(self._postcodedb)
                 X_price = X_price.dropna(subset=['medianPrice'])
                 y_price = X_price['medianPrice']
                 X = X_price.drop(['medianPrice','postcode'], axis=1)
                 if model == 'Random_Forest_house':
                    if update_hyperparameters:
                         param_gridRF = {
                            'n_estimators': [10,50,100],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                         }
                         reg = GridSearchCV(RandomForestRegressor(), param_gridRF, cv=5)
                         reg.fit(X, y_price)
                         self._models[model] = reg.best_estimator_
                    else:
                         reg = RandomForestRegressor(**Parameters_house)
                         reg.fit(X, y_price)
                         self._models[model] = reg 
            if model in local_authority_methods:
                X_local = self.preprocess_data_local(self._postcodedb)
                y_local = X_local['localAuthority']
                X = X_local.drop(['localAuthority','postcode'], axis=1)
                if model == 'Random_Forest_local':
                    if update_hyperparameters:
                        param_gridRF = {
                            'n_estimators': [10,50,100],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                        }
                        reg = GridSearchCV(RandomForestClassifier(), param_gridRF, cv=5)
                        reg.fit(X, y_local)
                        self._models[model] = reg.best_estimator_
                    else:
                        reg = RandomForestClassifier()
                        reg.fit(X, y_local)
                        self._models[model] = reg
            if model in historic_flooding_methods:
                X_hist = self.preprocess_hist(self._postcodedb)
                y_hist = X_hist['historicallyFlooded']
                X = X_hist.drop(['historicallyFlooded','postcode'], axis=1)
                if model == 'Random_Forest_hist':
                    if update_hyperparameters:
                        param_gridRF = {
                            'n_estimators': [10,50,100],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                        }
                        reg = GridSearchCV(RandomForestClassifier(), param_gridRF, cv=5)
                        reg.fit(X, y_hist)
                        self._models[model] = reg.best_estimator_
                    else:
                        reg = RandomForestClassifier()
                        reg.fit(X, y_hist)
                        self._models[model] = reg                                     

    def lookup_easting_northing(self, postcodes: Sequence,
                                dtype: np.dtype = np.float64) -> pd.DataFrame:
        '''Get a dataframe of OS eastings and northings from a sequence of
        input postcodes in the labelled or unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the easting and northing columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of 'easthing' and 'northing',
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the available postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['RH16 2QE'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        >>> results = tool.lookup_easting_northing(['RH16 2QE', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        AB1 2PQ        NaN       NaN
        '''

        postcodes = pd.Index(postcodes)

        frame = self._postcodedb.copy()
        frame = frame.set_index('postcode')
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ['easting', 'northing']].astype(dtype)

    def lookup_lat_long(self, postcodes: Sequence,
                        dtype: np.dtype = np.float64) -> pd.DataFrame:
        '''Get a Pandas dataframe containing GPS latitude and longitude
        information for a sequence of postcodes in the labelled or
        unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the latitude and longitude columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NaNs in the latitude
            and longitude columns.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        '''

        postcodes = pd.Index(postcodes)
        columns = ['postcode','easting', 'northing']
        labelled_data = self._postcodedb[columns]
        unlabelled_data = self._unlabelled_data[columns]
        combined_data = pd.concat([labelled_data, unlabelled_data])
        combined_data.drop_duplicates(subset='postcode', keep='first', inplace=True)
        combined_data.set_index('postcode', inplace=True)
        combined_data = combined_data.reindex(postcodes)
        lat, long = get_gps_lat_long_from_easting_northing(
            combined_data['easting'].values, combined_data['northing'].values, dtype=dtype
            )
        combined_data['latitude'] = lat
        combined_data['longitude'] = long
        return combined_data.loc[postcodes, ['latitude', 'longitude']].astype(dtype)

    def impute_missing_values(self, dataframe: pd.DataFrame,
                              method: str = 'mean',
                              constant_values: dict = IMPUTATION_CONSTANTS
                              ) -> pd.DataFrame:
        '''Impute missing values in a dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of the unlabelled postcode data)
            potentially containing missing values as NaNs, or with missing
            columns.

        method : str, optional
            Method to use for imputation. Options include:
            - 'mean', to use the mean for the labelled dataset
            - 'constant', to use a constant value for imputation
            - 'knn' to use k-nearest neighbours imputation from the
              labelled dataset

        constant_values : dict, optional
            Dictionary containing constant values to
            use for imputation in the format {column_name: value}.
            Only used if method is 'constant'.

        Returns
        -------

        pandas.DataFrame
            DataFrame with missing values imputed.

        Examples
        --------

        >>> tool = Tool()
        >>> missing = os.path.join(_example_dir, 'postcodes_missing_data.csv')
        >>> data = pd.read_csv(missing)
        >>> data = tool.impute_missing_values(data)  # doctest: +SKIP
        '''
        df = dataframe.copy()

        if method in ['mean', 'most_frequent']:
            imputer = SimpleImputer(strategy=method)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
        elif method == 'constant':
            for col, value in constant_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_help = df.select_dtypes(include=['float64', 'int64'])
            df_values = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
            df[df.select_dtypes(include=['object']).columns] = df_values
        else:
            raise ValueError(f'Unknown imputation method: {method}')
    
        return df

    def predict_flood_class_from_postcode(self, postcodes: Sequence[str]
                                         
                                          ) -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
            Returns NaN for postcode units not in the available postcode files.
        '''
        
        df = self._postcodedb.copy()
        df = df[df['postcode'].isin(postcodes)].set_index('postcode')
        df = self.preprocesss_data_flood(df)
        X = df.drop(['postcode','riskLabel'], axis=1, errors='ignore')
        y_pred = self._models['Random_Forest_flood'].predict(X)
        return pd.Series(
            data=y_pred.astype(int),
            index=df.index,
            name='riskLabel'
        )

    def predict_flood_class_from_OSGB36_location(
            self, eastings: Sequence[float], northings: Sequence[float],
            method: str = 'all_zero_risk') -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        '''

        if method not in ['Random_Forest_flood']:
            raise NotImplementedError(f"Method '{method}' not implemented")

        df = pd.DataFrame({
            'easting': eastings,
            'northing': northings
        })
        merged_data = pd.merge(df, self._postcodedb, on=['easting', 'northing'], how='inner')
        df = self.preprocesss_data_flood(merged_data)
        X = df.drop(['postcode','riskLabel'], axis=1, errors='ignore')
        y_pred = self._models[method].predict(X)
        return pd.Series(
            data=y_pred.astype(int),
            index=df.index,
            name='riskLabel'
        )
    
    def predict_flood_class_from_WGS84_locations(
            self, longitudes: Sequence[float], latitudes: Sequence[float],
            method: str = 'all_zero_risk') -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels multi-indexed by
            location as a (longitude, latitude) pair.
        '''

        if method == 'all_zero_risk':
            idx = pd.MultiIndex.from_tuples([(lng, lat) for lng, lat in
                                             zip(longitudes, latitudes)])
            return pd.Series(
                data=np.ones(len(longitudes), int),
                index=idx,
                name='riskLabel',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_median_house_price(
            self, postcodes: Sequence[str],
            method: str = 'all_england_median'
            ) -> pd.Series:
        '''
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        '''

        if method not in house_price_methods:
            raise NotImplementedError(f'method {method} not implemented')
        df = self._postcodedb.copy()
        df = df[df['postcode'].isin(postcodes)].set_index('postcode')
        df = self.preprocess_data_house(df)
        X = df.drop(['postcode','medianPrice'], axis=1, errors='ignore')
        y_pred = self._models[method].predict(X)
        prediction_series = pd.Series(data=y_pred, index=postcodes, name='medianPrice')
        full_prediction_series = prediction_series.reindex(postcodes)
        return full_prediction_series   

    def predict_local_authority(
        self, eastings: Sequence[float], northings: Sequence[float],
        method: str = 'do_nothing'
    ) -> pd.Series:
        '''
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            locations, and multiindexed by the location as a
            (easting, northing) tuple.
        '''

        if method not in local_authority_methods:
            raise NotImplementedError(f'method {method} not implemented')
        
        df = pd.DataFrame({
            'easting': eastings,
            'northing': northings
        })
        merged_data = pd.merge(df, self._postcodedb, on=['easting', 'northing'], how='inner')
        df = self.preprocess_data_local(merged_data)
        X = df.drop(['postcode','localAuthority'], axis=1, errors='ignore')
        y_pred = self._models[method].predict(X)
        tuple_index = pd.MultiIndex.from_tuples(list(zip(eastings, northings)), names=["easting", "northing"])
        return pd.Series(
            data=y_pred,
            index=tuple_index,
            name='localAuthority'
        )
        

    def predict_historic_flooding(
            self, postcodes: Sequence[str],
            method: str = 'all_false'
            ) -> pd.Series:
        '''
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        '''
        if method not in ['Random_Forest_hist']:
            raise NotImplementedError(f'method {method} not implemented')
        df = self._postcodedb.copy()
        df = df[df['postcode'].isin(postcodes)].set_index('postcode')
        df = self.preprocess_hist(df)
        X = df.drop(['postcode','historicallyFlooded'], axis=1, errors='ignore')
        y_pred = self._models[method].predict(X)
        return pd.Series(
            data=y_pred.astype(int),
            index=df.index,
            name='riskLabel'
        )

        

    def estimate_total_value(self, postal_data: Sequence[str]) -> pd.Series:
        '''
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        The estimate is based on the median house price for the area and an
        estimate of the number of properties it contains.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcode sectors (either
            may be used).


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        '''
        def is_sector(postcode: str) -> bool:
            """
            Determine if a given postcode is a sector or a unit.

            Parameters:
            - postcode: str

            Returns:
            - bool: True if sector, False if unit
            """
            parts = postcode.strip().split()
            if len(parts) == 1:
                return True 
            elif len(parts) == 2 and len(parts[1]) == 3:
                return False  # Likely a unit
            else:
                return len(parts[1]) == 3

       
        sector_map = {}
        for pc in postal_data:
            pc = pc.upper().strip()
            if is_sector(pc):
                sector = pc
            else:
                sector = pc.split()[0]
            sector_map[pc] = sector
        mapping_df = pd.DataFrame.from_dict(sector_map, orient='index', columns=['sector'])
        mapping_df.index.name = 'postal_code'
        mapping_df.reset_index(inplace=True)

        merged_df = mapping_df.merge(
            self._sector_data,
            left_on='sector',
            right_on='postcodeSector',
            how='left'
        )
        merged_df = merged_df.merge(
            self.median_house_price.rename('median_house_price'),
            left_on='sector',
            right_index=True,
            how='left'
        )

        missing_prices = merged_df[merged_df['median_house_price'].isnull()]['sector'].unique()
        if len(missing_prices) > 0:
            raise ValueError(f"Missing median house price data for sectors: {missing_prices}")

        merged_df['total_property_value'] = merged_df['households'] * merged_df['median_house_price']

        result = merged_df.set_index('postal_code')['total_property_value']

        return result

    def estimate_annual_human_flood_risk(self, postcodes: Sequence[str],
                                         risk_labels: [pd.Series | None] = None
                                         ) -> pd.Series:
        '''
        Return a series of estimates of the risk to human life for a
        collection of postcodes.

        Risk is defined here as an impact coefficient multiplied by the
        estimated number of people under threat multiplied by the probability
        of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual human flood risk estimates
            indexed by postcode.
        '''

        if risk_labels is None:
            risk_labels = self.predict_flood_class_from_postcode(postcodes)
        else:
            risk_labels = risk_labels

        postcodes_ = [split_postcode(postcode) for postcode in postcodes]


        flood_probs = [0.001,0.002,0.005,0.01,0.02,0.03,0.05]

        rows = []

        for i, postcode in enumerate(postcodes_):

            #count total number of postcodes from sector data numberOfPostcodeUnits for a given postcode
            count = self._sector_data[self._sector_data['postcodeSector'] == postcode]['numberOfPostcodeUnits']

            population = self._sector_data[self._sector_data['postcodeSector'] == postcode]['headcount'] / count
            
            total_human_value = 0.1*population*flood_probs[risk_labels[i]-1]

            rows.append({'postcode': postcodes[i], 'total_esitimated_human_value': total_human_value})


        return pd.DataFrame(rows)

    def estimate_annual_flood_economic_risk(
            self, postcodes: Sequence[str],
            risk_labels: [pd.Series | None] = None
            ) -> pd.Series:
        '''
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            optionally provide a Pandas Series containing flood risk
            classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual economic flood risk estimates indexed
            by postcode.
        '''

        if risk_labels is None:
            lood_risks = self.predict_flood_class_from_postcode(postcodes)
        else:
            flood_risks = risk_labels


        #split post codes into outcode and  first digit of incode eg BA1 3PD -> BA1 3
        postcodes_ = [split_postcode(postcode) for postcode in postcodes]


        flood_probs = [0.001,0.002,0.005,0.01,0.02,0.03,0.05]

        rows = []


        for i, postcode in enumerate(postcodes_):

            count = self._sector_data[self._sector_data['postcodeSector'] == postcode]['numberOfPostcodeUnits']

            #find median house price and number of properties in postcode

            median_price = self.predict_median_house_price(postcodes[i])
            num_properties = self._sector_data[self._sector_data['postcode'] == postcode]['households'] / count

            #calculate total value using 0.05*totalpropertyvalue*probability of flooding

            total_value = 0.05*median_price*num_properties*flood_probs[flood_risks[i]-1]

            #save to df and return
            
            rows.append({'postcode': postcodes[i], 'total_estimated_value': total_value})

        return pd.DataFrame(rows)
