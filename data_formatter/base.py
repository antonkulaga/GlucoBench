'''Defines a generic data formatter for CGM data sets.'''
import warnings
import numpy as np
import pandas as pd
import sklearn.preprocessing
import data_formatter.types as types
import data_formatter.utils as utils

DataTypes = types.DataTypes
InputTypes = types.InputTypes

dict_data_type = {'categorical': DataTypes.CATEGORICAL,
                  'real_valued': DataTypes.REAL_VALUED,
                  'date': DataTypes.DATE}
dict_input_type = {'target': InputTypes.TARGET,
                   'observed_input': InputTypes.OBSERVED_INPUT,      
                   'known_input': InputTypes.KNOWN_INPUT,
                   'static_input': InputTypes.STATIC_INPUT,
                   'id': InputTypes.ID,
                   'time': InputTypes.TIME}


class DataFormatter():
  # Defines and formats data for the IGLU dataset.

  def __init__(self, cnf):
    """Initialises formatter."""
    # load parameters from the config file
    self.params = cnf
    
    # load column definition
    print('-'*32)
    print('Loading column definition...')
    self.__process_column_definition()

    # check that column definition is valid
    print('Checking column definition...')
    self.__check_column_definition()

    # load data
    # check if data table has index col: -1 if not, index >= 0 if yes
    print('Loading data...')
    self.params['index_col'] = False if self.params['index_col'] == -1 else self.params['index_col']
    # read data table
    self.data = pd.read_csv(self.params['data_csv_path'], index_col=self.params['index_col'], na_filter=False)

    # check NA values
    print('Checking for NA values...')
    self.__check_nan()

    # set data types in DataFrame to match column definition
    print('Setting data types...')
    self.__set_data_types()

    # check time grid
    print('Checking time grid...')
    self.__check_time_grid()

    # drop columns / rows
    print('Dropping columns / rows...')
    self.__drop()

    # encode
    print('Encoding data...')
    self._encoding_params = self.params['encoding_params']
    self.__encode()

    # interpolate
    print('Interpolating data...')
    self._interpolation_params = self.params['interpolation_params']
    self._interpolation_params['interval_length'] = self.params['observation_interval']
    self.__interpolate()

    # split data
    print('Splitting data...')
    self._split_params = self.params['split_params']
    self.__split_data()

    # scale
    print('Scaling data...')
    self._scaling_params = self.params['scaling_params']
    self.__scale()

    print('Data formatting complete.')
    print('-'*32)

  def __process_column_definition(self):
    self._column_definition = []
    for col in self.params['column_definition']:
      self._column_definition.append((col['name'], 
                                      dict_data_type[col['data_type']], 
                                      dict_input_type[col['input_type']]))

  def __check_column_definition(self):
    # check that there is unique ID column
    assert len([col for col in self._column_definition if col[2] == InputTypes.ID]) == 1, 'There must be exactly one ID column.'
    # check that there is unique time column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TIME]) == 1, 'There must be exactly one time column.'
    # check that there is at least one target column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TARGET]) >= 1, 'There must be at least one target column.'
  
  def __set_data_types(self):
    # set time column as datetime format in pandas
    for col in self._column_definition:
      if col[1] == DataTypes.DATE:
        self.data[col[0]] = pd.to_datetime(self.data[col[0]])
      if col[1] == DataTypes.CATEGORICAL:
        self.data[col[0]] = self.data[col[0]].astype('category')
      if col[1] == DataTypes.REAL_VALUED:
        self.data[col[0]] = self.data[col[0]].astype('float')

  def __check_nan(self):
    if self.params['nan_vals'] is not None:
      # replace NA values with pd.np.nan
      self.data = self.data.replace(self.params['nan_vals'], np.nan)
    # delete rows where target, time, or id are na
    self.data = self.data.dropna(subset=[col[0] 
                                  for col in self._column_definition 
                                  if col[2] in [InputTypes.TARGET, InputTypes.TIME, InputTypes.ID]])

  def __check_time_grid(self):
    # get time column
    time_col_name = [col[0] for col in self._column_definition if col[2] == InputTypes.TIME][0]
    time_col = self.data[time_col_name]
    # round time to minutes
    time_col = time_col.dt.round('min')
    # compute gaps between time points
    time_gaps = time_col.diff().dt.total_seconds().fillna(0)
    # convert str indicating observation_interval to seconds 
    observation_interval = pd.Timedelta(self.params['observation_interval']).total_seconds()
    # if time gaps are within observational_interval + 50%, assume one segment, otherwise assume next segment
    segments = (time_gaps > observation_interval * 1.5).cumsum()
    # iterate over segments
    num_deviating_gaps = 0
    for segment in segments.unique():
      # get time points in segment
      segment_time = time_col[segments == segment]
      # compute time gaps
      segment_time_gaps = segment_time.diff().dt.total_seconds().fillna(0)
      # check if time gaps are equal to observation interval, otherwise warn
      if not np.allclose(segment_time_gaps, observation_interval):
        num_deviating_gaps += 1
      # fix time
      segment_time = pd.date_range(start=segment_time.min(),
                                       periods=len(segment_time),
                                       freq=self.params['observation_interval'])
      # set time column to fixed time
      time_col[segments == segment] = segment_time
    if num_deviating_gaps > 0:
      self.data[time_col_name] = time_col
      print('\tWARNING: {} time gaps deviate from observation interval.'.format(num_deviating_gaps))


  def __drop(self):
    # drop columns that are not in the column definition
    self.data = self.data[[col[0] for col in self._column_definition]]
    # drop rows based on conditions set in the formatter
    if self.params['drop'] is not None:
      for col in self.params['drop'].keys():
        self.data = self.data.loc[~self.data[col].isin(self.params['drop'][col])].copy()
  
  def __interpolate(self):
    self.data, self._column_definition = utils.interpolate(self.data, 
                                                           self._column_definition, 
                                                           **self._interpolation_params)

  def __split_data(self):
    self.train_idx, self.val_idx, self.test_idx = utils.split(self.data, 
                                                              self._column_definition, 
                                                              **self._split_params)
    self.train_data, self.val_data, self.test_data = self.data.iloc[self.train_idx], self.data.iloc[self.val_idx], self.data.iloc[self.test_idx]

  def __encode(self):
    self.data, self._column_definition, self.encoders = utils.encode(self.data, 
                                                                     self._column_definition,
                                                                     **self._encoding_params)
  
  def __scale(self):
    self.train_data, self.val_data, self.test_data, self.scalers = utils.scale(self.train_data, 
                                                                               self.val_data, 
                                                                               self.test_data, 
                                                                               self._column_definition, 
                                                                               **self.params['scaling_params'])

  def get_column(self, column_name):
    # write cases for time, id, target, future, static, dynamic covariates
    if column_name == 'time':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.TIME][0]
    elif column_name == 'id':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.ID][0]
    elif column_name == 'sid':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.SID][0]
    elif column_name == 'target':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.TARGET]
    elif column_name == 'future_covs':
      future_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.KNOWN_INPUT] 
      return future_covs if len(future_covs) > 0 else None
    elif column_name == 'static_covs':
      static_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.STATIC_INPUT]
      return static_covs if len(static_covs) > 0 else None
    elif column_name == 'dynamic_covs':
      dynamic_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.OBSERVED_INPUT]
      return dynamic_covs if len(dynamic_covs) > 0 else None
    else:
      raise ValueError('Column {} not found.'.format(column_name))
  
