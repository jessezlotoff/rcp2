from src.data.LoadAndCleanACS import LoadAndCleanACS
from src.data.LoadAndCleanARCP import LoadAndCleanARCP
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

GEO_LEVEL_DICT = {'State': 2, 'County': 5, 'Tract': 11, 'Block': 12}
MAIN_PATH = Path.cwd().parent.parent

def CreateConfidenceIntervals(num_surveys, percentage):
    """ adds a confidence interval to clean data

    :param num_surveys: int
    :param percentage: float

    :returns: float
    """

    z =	1.960 # corresponds to 95% confidence interval
    
    CI =  z * np.sqrt((percentage * (100 - percentage) ) / num_surveys)

    return CI
    

def CreateSingleLevelData(geo_level, arcp_data, acs_data):
    """
    This function takes the arc data  into a dataset containing the percentage 
    and number of smoke detectors by census geography
    
    The resultant dataset will have the following values:
      num_surveys - total number of surveys conducted    
      detectors_found -   houses with at least one smoke detector in the home
      detectors_working - houses with at least one tested and working smoke 
        detector in the home
    
      Note: for variables the suffixes 
        _total- indicates raw counts 
        _prc  - indicates percentage: (_total / num_surveys * 100)

    :param geo_level: str var indcating what census geography to aggregate on
        State, County, Tract, Block
    :param arcp_data: pandas dataframe with ARC Preparedness data
    :param acs_data: pandas dataframe with ACS data

    :returns: pandas dataframe
    """

    arcp_data['geoid'] = arcp_data['geoid'].str[: GEO_LEVEL_DICT[geo_level]]
    
    acs.index =  acs_data.index.str[:GEO_LEVEL_DICT[geo_level]]
    acs_data.drop_duplicates(inplace = True)
    
    ## binarize pre_existing_alarms and _tested_and_working
    #  values will now be: 0 if no detectors present and 1 if any number were 
    # present
    arcp_data['pre_existing_alarms'].where(arcp_data['pre_existing_alarms'] < 1, 
                                           other = 1, inplace = True) 
    arcp_data['pre_existing_alarms_tested_and_working']\
        .where(arcp_data['pre_existing_alarms_tested_and_working'] < 1,
               other = 1, inplace = True)

    ## create detectors dataset
    # This happens by grouping data both on pre_existing alarms and then 
    # _tested_and working alarms and then merging the two into the final dataset

    detectors =  arcp_data.groupby('geoid')['pre_existing_alarms']\
        .agg({np.size, np.sum, lambda x: np.sum(x)/np.size(x)* 100 })

    detectors.rename({'size':'num_surveys','sum':'detectors_found_total',
                      '<lambda_0>':'detectors_found_prc'},
                     axis =1, inplace = True)

    detectors['detectors_found_prc'] = detectors['detectors_found_prc'].round(2)
    
  
    
    d2 =  arcp_data.groupby('geoid')['pre_existing_alarms_tested_and_working']\
            .agg({np.size,np.sum, lambda x: np.sum(x)/np.size(x)* 100})
    
    d2.rename({'size':'num_surveys2','sum':'detectors_working_total',
               '<lambda_0>':'detectors_working_prc'},
               axis =1, inplace = True)

    d2['detectors_working_prc'] = d2['detectors_working_prc'].round(2)
    

    detectors = detectors.merge(d2,how = 'left', on ='geoid')

    detectors['detectors_found_CI'] = \
        CreateConfidenceIntervals(detectors['num_surveys'].values,
                                  detectors['detectors_found_prc'].values)
                                                                
    detectors['detectors_working_CI'] = \
        CreateConfidenceIntervals(detectors['num_surveys'].values,
                                  detectors['detectors_working_prc'].values)  
    
    # rearrange columns 
    column_order = ['num_surveys',	
                    'detectors_found_total',
                    'detectors_found_prc', 
                    'detectors_found_CI',
                    'detectors_working_total',
                    'detectors_working_prc',
                    'detectors_working_CI']
    
    detectors = detectors[column_order]
    
    detectors = detectors[~pd.isna(detectors.index)]
    
    # ensure blocks that weren't visited are added to the model 
    detectors = detectors.reindex(detectors.index.union(acs_data.index.unique()),
                                  fill_value = 0)
    detectors = detectors[~pd.isna(detectors.index)]
   
    return detectors


def ExportSingleLevelData(model_data, geo_level):
    """ export single level model results to csv

    :param model_data: pandas dataframe with model results
    :param geo_level: str var indcating what census geography to aggregate on

    :returns: None
    """

    model_data = model_data.copy()
    filename = f"SmokeAlarmModel{geo_level}.csv"
    model_data.index.name = 'geoid'
    model_data.index =  '#_' + model_data.index 
    out_path =  MAIN_PATH / 'Data' / 'Model Outputs' / 'Smoke_Alarm_Single_Level' 
    output_path = output_path / filename
    model_data.to_csv(out_path)


def CreateMultiLevelData(state_data, county_data, tract_data, block_data):
    """ combine single level models, filling in blanks with larger geographies

    :param state_data: pandas dataframe from single level State model
    :param county_data: pandas dataframe from single level County model
    :param tract_data: pandas dataframe from single level Tract model
    :param block_data: pandas dataframe from single level Block model

    :returns: pandas dataframe
    """

    # start with block level data
    all_IDS = block_data.index
    block_data = block_data[block_data['num_surveys'] >= 30]
    block_data['geography'] = 'block'
    block_data.index.name = 'geoid'

    remaining_ids = all_IDS[~all_IDS.isin(block_data.index)]
    remaining_ids = remaining_ids.to_frame()
    remaining_ids = remaining_ids.rename({0:'geoid'}, axis = 1)

    MultiLevelModel = block_data

    # fill in missing geographies via tract, county, state
    geo_list = [
        ('Tract', GEO_LEVEL_DICT['Tract'], tract_data),
        ('County', GEO_LEVEL_DICT['County'], county_data),
        ('State', GEO_LEVEL_DICT['State'], state_data)
    ]

    for  geo, geo_len, df in geo_list:

        # find all remaining ids that are not in the block data     
        geo_index = remaining_ids

        # set up data index 
        geo_index['temp_geoid'] = geo_index.index.str[:geo_len]
        geo_index = geo_index.set_index('geoid')
        
        # create data set at one level
        geo_data = geo_index.merge(df, how='left', right_index=True, 
                                   left_on='temp_geoid')
        geo_data = geo_data[geo_data['num_surveys'] > 30] 
        geo_data = geo_data.drop('temp_geoid',axis = 1 )
        geo_data['geography'] = geo
        
        # add to multilevel index
        MultiLevelModel = MultiLevelModel.append(geo_data)
        
        # update remaining_ids
        remaining_ids = remaining_ids[~remaining_ids.index.isin(MultiLevelModel.index)]
        del geo_index, geo_data

    MultiLevelModel = MultiLevelModel.reset_index()
    MultiLevelModel['geoid'] = '#_' + MultiLevelModel['geoid']

    out_path =  MAIN_PATH / 'Data' / 'Model Outputs' / 'SmokeAlarmModelOutput.csv'

    MultiLevelModel.to_csv(out_path)

    return MultiLevelModel


def PrepForModelTraining(comb_data, acs_data):

    comb_data = comb_data[['geoid', 'num_surveys','detectors_found_prc']]
    # comb_data['geoid'] = comb_data['geoid'].str[2:]
    comb_data = comb_data.set_index('geoid')
    Output_Var = comb_data[['detectors_found_prc']]
    acs_data = acs_data.drop(['house_pct_vacant', 'did_not_work_past_12_mo', 
                              'house_pct_non_family', 'house_pct_rent_occupied',
                              'race_pct_nonwhite', 'race_pct_nonwhitenh', 
                              'house_pct_incomplete_plumb', 
                              'house_pct_incomplete_kitchen', 
                              'race_pct_whitenh'], 
                              axis=1) 
    ACS_SmokeAlarm = acs_data[acs_data.index.isin(comb_data.index.tolist())]
    Data_Matrix = ACS_SmokeAlarm.merge(Output_Var, how='left', left_index=True, 
                                       right_index=True)
    X = Data_Matrix.loc[:, Data_Matrix.columns != 'detectors_found_prc']
    y = Data_Matrix.loc[:, Data_Matrix.columns == 'detectors_found_prc']

    return X, y


def PrepARCPData(arcp, acs):

    # split and recombine ARC data by geographic level for completeness
    state_data = CreateSingleLevelData('State', arcp.copy(), acs.copy())
    county_data = CreateSingleLevelData('County', arcp.copy(), acs.copy())
    tract_data = CreateSingleLevelData('Tract', arcp.copy(), acs.copy())
    block_data = CreateSingleLevelData('Block', arcp.copy(), acs.copy())
    comb_data = CreateMultiLevelData(state_data, county_data, tract_data, 
                                     block_data)

    return comb_data


def TrainModel(X, y):

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(X, 
                    y, test_size = 0.30, random_state = 0)
    scaler = preprocessing.StandardScaler().fit(train_features)
    scaler.transform(train_features)
    scaler.transform(test_features)

    rf = RandomForestRegressor(n_estimators = 40, random_state = 0)

    # Train the model on training data
    model=rf.fit(train_features, train_labels['detectors_found_prc'].ravel())
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(test_features.columns)):
        print("%d. %s (%f)" % (f + 1, test_features.columns[indices[f]], importances[indices[f]]))
        
    # Use the forest's predict method on the test data
    predictions = model.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels['detectors_found_prc'].ravel())
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    print('Median Absolute Error:', round(np.median(errors), 2))
    print('Standard Deviation of Absolute Error:', round(np.std(errors), 2))

    return model


def BuildSmokeAlarmModel():
    
    # load and preprocess local ACS and ARC Preparedness data
    acs = LoadAndCleanACS()
    arcp = LoadAndCleanARCP()

    comb_data = PrepARCPData(arcp, acs)
    X, y = PrepForModelTraining(comb_data, acs.copy())

    model = TrainModel(X, y)


if __name__ == "__main__":
    BuildSmokeAlarmModel()
    