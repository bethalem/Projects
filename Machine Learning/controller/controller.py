# # Machine Learning Predictive Model for Flight Pricing 
# ### DTSC 691 
# ### Machine Learning
# ### Project Proposal
# ### Bethlehem Alem 

# #### Background
# Traveling has become one of the biggest components of our lives to keep us connected to one another. There are different forms of transportation to travel to different places for different reasons. People travel for leisure, medical treatment, work, or other purposes by car, train, and, most importantly, air. Thousands of Americans travel by airplane every day. The federal aviation administration air route traffic control center reported that there are more than 674 million passengers in 2021 (BTS, 2022). For most people when considering flying by air it is important to get a fair price. There are many websites that tell us when to buy tickets to get a better-priced fare. These websites use AI to study all the data history, predict future rates, and show what day to buy tickets. 
# The Bureau of transportation statistics has a database for all transportation means in the USA. It contains data from military aviation to bike/pedestrian databases. It also contains subjects that influence traveling experiences from safety to energy to environmental factors. The database is well organized and labeled to download specific attributes the user wanted in xls or pdf format. Their website makes the latest available data accessible to the public in an organized and alphabetically sorted-out way.  

# #### General analytics
# This project aims to collect and analyze flight data to accurately predict prices for flights from various attributes. The recorded data is more than one million rows and will be reduced by more than half. The price and distance in miles, including ten other columns, were taken from the Origin and Destination survey data. The data were filtered for tickets administered by American Airlines for the 1st quarter of 2021. The data collection might be influenced by other factors like covid 19, But it will show us the best values in the past year. 
# For this project, Jupyter notebook will be utilized for the majority of the data analyses.  A summary of descriptive statistics will be performed to identify key attributes that affect flight prices. Then, various ML predictive models will be implemented to discover a preferable algorithm that works with the data. Finally, a grid search will be performed to hyper-tune the models and increase accuracy.

# Common imports
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

class controller():
    def __init__(self) -> None:
        self.col_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler()
        self.fs = SelectKBest(score_func=f_regression, k="all")
        pass    

    def save_fig(self, IMAGES_PATH, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("\tSaving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
    
    def display_scores(self, scores):
        print("\tCV Top Score:", scores[0])
        print("\tMean Error:", scores.mean())
        print("\tStandard deviation:", scores.std())
        
    def encode_test(self, test_set):
        ################################################ Testing Set
        test_set.set_index([pd.Index(range(0, test_set.shape[0]))], inplace=True)
        
        # seeing categorical columns    
        cols = test_set.columns
        num_cols = test_set._get_numeric_data().columns
        cat_columns = list(set(cols) - set(num_cols))
        
        # seeing numerical columns    
        num_columns = list(num_cols)
        
        # creating categoical encoded data
        cols_data = self.col_enc.transform(test_set[cat_columns])
        test_set.drop(cat_columns, axis = 1, inplace = True)
        cols_data = pd.DataFrame(cols_data)
        test_set = pd.concat([test_set, cols_data], axis= 1)
        
        # creating numerical encoded data    
        scaled = self.scaler.transform(test_set[num_columns])
        test_set.drop(num_columns, axis = 1, inplace = True)
        scaled = pd.DataFrame(scaled, columns = range(len(test_set.columns), len(test_set.columns)+scaled.shape[1]))
        test_set = pd.concat([test_set, scaled], axis= 1)
        return test_set    
        
    def encode_train(self, train_set):
        train_set.set_index([pd.Index(range(0, train_set.shape[0]))], inplace=True)

        ######################################### Training set
        # seeing categorical columns    
        cols = train_set.columns
        num_cols = train_set._get_numeric_data().columns
        cat_columns = list(set(cols) - set(num_cols))
        
        # seeing numerical columns    
        num_columns = list(num_cols)
        
        # creating categoical encoded data
        cols_data = self.col_enc.fit_transform(train_set[cat_columns])
        train_set.drop(cat_columns, axis = 1, inplace = True)
        cols_data = pd.DataFrame(cols_data)
        train_set = pd.concat([train_set, cols_data], axis= 1)
        
        # creating numerical encoded data    
        scaled = self.scaler.fit_transform(train_set[num_columns])
        train_set.drop(num_columns, axis = 1, inplace = True)
        scaled = pd.DataFrame(scaled, columns = range(len(train_set.columns), len(train_set.columns)+scaled.shape[1]))
        train_set = pd.concat([train_set, scaled], axis= 1)
        return train_set    
    
    def select_features_train(self, train_set, train_set_label):
        self.fs.fit(train_set, train_set_label)
        train_set_fs = self.fs.transform(train_set)
        return train_set_fs

    def select_features_test(self, test_set):
        train_set_fs = self.fs.transform(test_set)
        return train_set_fs
        
    def run(self, test_set):
        matplotlib.rc('axes', labelsize=14)
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)

        PROJECT_ROOT_DIR = "."
        IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "static/visualizations")
        os.makedirs(IMAGES_PATH, exist_ok=True)

        # Read in Data from excel file
        flight_info =pd.read_csv("dataset/Origin_and_Destination_Survey_DB1BMarket_2021_1.csv")
        # flight_info.drop(flight_info.head(1047000).index, inplace = True)
        np.random.seed(10)


        ### If you would like to run the whole dataset, no need to run this block of code #############

        ############
        remove_n = 1047575
        drop_indices = np.random.choice(flight_info.index, remove_n, replace=False)
        flight_info = flight_info.drop(drop_indices)
        pd.set_option('display.max_columns', None)

        ##############

        data = flight_info.copy()

        # convert all coloumns with type float to type int except MKTFare
        data.loc[:, ['ItinID', 'MktID']] = data.loc[:, ['ItinID', 'MktID']].astype(np.int64)

        # 1. Drop unnessasary columns
        # 2. reduce the data by selecting important ticket carriers
        # 3. drop Na 
        train_set = data.drop(['ItinID', 'MktID', 
                            'Year','Quarter',
                            'OriginCountry', 'OriginAirportID', 'OriginCityMarketID','Origin','OriginStateFips','OriginState', 'OriginWac',
                            'DestCountry', 'DestAirportID', 'DestCityMarketID', 'Dest', 'DestStateFips', 'DestState', 'DestWac',
                            'WacGroup', 'TkCarrierChange', 'TkCarrierGroup', 'OpCarrierChange', 'OpCarrierGroup',
                            'RPCarrier', 'OpCarrier', 'BulkFare', 'MktDistanceGroup', 'MktMilesFlown', 'NonStopMiles', 'MktGeoType'], axis=1)

        train_set.dropna()

        train_set[['Airport1','Airport2','Airport3']] = train_set['AirportGroup'].str.split(':', 2, expand=True)
        del train_set['AirportGroup']

        train_set_label = train_set["MktFare"].copy()
        del train_set['MktFare']

        print("\n[LOG]: Saving images to images folder")
        
        train_set.groupby('TkCarrier').MktDistance.sum().plot(kind='pie',figsize=(16,8))
        self.save_fig(IMAGES_PATH, "TkCarrier")
        
        plt.style.use('ggplot')
        train_set.groupby('OriginStateName').MktDistance.sum().plot(kind='bar',figsize=(16,8))
        self.save_fig(IMAGES_PATH, "CountOriginState")
        
        train_set.groupby('DestStateName').MktDistance.sum().plot(kind='bar',figsize=(16,8))
        self.save_fig(IMAGES_PATH, "CountDestState")
        
        train_set.hist(bins=50, figsize=(20,15))
        self.save_fig(IMAGES_PATH, "AttributeHistogramPlots")

        train_set = train_set.drop(['MktDistance'], axis = 1)

        train_set = self.encode_train(train_set)
        train_set = self.select_features_train(train_set, train_set_label)
        train_set = self.select_features_train(train_set, train_set_label)

        lin_reg = LinearRegression()
        print("\n[LOG]: Fitting linear regression")
        lin_reg.fit(train_set, train_set_label)

        flight_pricing_predictions = lin_reg.predict(train_set)
        
        lin_mse = mean_squared_error(train_set_label, flight_pricing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        
        lin_mae = mean_absolute_error(train_set_label, flight_pricing_predictions)
        print("\tMSE:", lin_rmse)
        print("\tMAE:", lin_mae)
        
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(train_set, train_set_label)

        flight_pricing_predictions = tree_reg.predict(train_set)
        tree_mse = mean_squared_error(train_set_label, flight_pricing_predictions)
        tree_rmse = np.sqrt(tree_mse)

        lin_mae = mean_absolute_error(train_set_label, flight_pricing_predictions)



        print("\n[LOG]: Fitting Tree regression")
        scores = cross_val_score(tree_reg, train_set, train_set_label, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)


        param_grid = {'splitter': ['best', 'random'], 'min_samples_split': [2, 5, 8, 14, 16, 18, 20],
                      'max_depth': [1, 3, 4, 5, 8, 16, 32]}

        grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, verbose=1, cv=3)
        grid_search_cv.fit(train_set, train_set_label)  # fit the training data
        print("The best parameters are: ", grid_search_cv.best_params_)


        final_model = grid_search_cv.best_estimator_

        final_model.fit(train_set, train_set_label)
        flight_pricing_predictions = final_model.predict(train_set)
        tree_mse = mean_squared_error(train_set_label, flight_pricing_predictions)
        tree_rmse = np.sqrt(tree_mse)

        lin_mae = mean_absolute_error(train_set_label, flight_pricing_predictions)
        
        test_set = self.encode_test(test_set)
        test_set = self.select_features_test(test_set)
        
        print()
        
        final_predictions = final_model.predict(test_set)
        return final_predictions[0]