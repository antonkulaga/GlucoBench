import sys
import os
import pickle
import gzip
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

import darts
from darts import metrics

from lib.gluformer.model import *
from lib.latent_ode.trainer_glunet import *
from utils.darts_processing import *
from utils.darts_dataset import *

# Ensure directories for saving results exist
os.makedirs('./output/data', exist_ok=True)
os.makedirs('./output/plots', exist_ok=True)

# MODELS: TRANSFORMER, NHiTS, XGBOOST, LINEAR REGRESSION

# model params
model_params = {
    'transformer': {'darts': models.TransformerModel, 'darts_data': SamplingDatasetInferencePast, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'nhits': {'darts': models.NHiTSModel, 'darts_data': SamplingDatasetInferencePast, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'xgboost': {'darts': models.XGBModel, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'linreg': {'darts': models.LinearRegressionModel, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'}
}

# data sets
datasets = ['livia_mini']
save_trues = {}
save_forecasts = {}
save_inputs = {}

# iterate through models and datasets
for model_name in model_params.keys():
    for dataset in datasets:
        print(f'Testing {model_name} for {dataset}')
        formatter, series, scalers = load_data(seed=0, study_file=None, dataset=dataset, 
                                               use_covs=model_params[model_name]['use_covs'], 
                                               use_static_covs=model_params[model_name]['use_static_covs'],
                                               cov_type=model_params[model_name]['cov_type'])
        # load model or refit model
        if model_name in ['transformer', 'nhits']:
            # load model
            model = model_params[model_name]['darts'](input_chunk_length=formatter.params[model_name]['in_len'],
                                                      output_chunk_length=formatter.params['length_pred'])
            model = model.load_from_checkpoint(f'tensorboard_{model_name}_{dataset}', work_dir='./output', best=True)
            # define dataset for inference
            test_dataset = model_params[model_name]['darts_data'](target_series=series['test']['target'],
                                                                  n=formatter.params['length_pred'],
                                                                  input_chunk_length=formatter.params[model_name]['in_len'],
                                                                  output_chunk_length=formatter.params['length_pred'],
                                                                  use_static_covariates=model_params[model_name]['use_static_covs'],
                                                                  max_samples_per_ts=None)
            # get predictions
            forecasts = model.predict_from_dataset(n=formatter.params['length_pred'], 
                                                   input_series_dataset=test_dataset,
                                                   verbose=True,
                                                   num_samples=20 if model_name == 'tft' else 1)
            forecasts = scalers['target'].inverse_transform(forecasts)
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = [test_dataset.evalsample(i) for i in range(len(test_dataset))]
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(save_trues[f'{model_name}_{dataset}'])
            # get inputs
            inputs = [test_dataset[i][0] for i in range(len(test_dataset))]
            save_inputs[f'{model_name}_{dataset}'] = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_

        elif model_name == 'xgboost':
            # load and fit model
            model = model_params[model_name]['darts'](lags=formatter.params[model_name]['in_len'], 
                                                      learning_rate=formatter.params[model_name]['lr'],
                                                      subsample=formatter.params[model_name]['subsample'],
                                                      min_child_weight=formatter.params[model_name]['min_child_weight'],
                                                      colsample_bytree=formatter.params[model_name]['colsample_bytree'],
                                                      max_depth=formatter.params[model_name]['max_depth'],
                                                      gamma=formatter.params[model_name]['gamma'],
                                                      reg_alpha=formatter.params[model_name]['alpha'],
                                                      reg_lambda=formatter.params[model_name]['lambda_'],
                                                      n_estimators=formatter.params[model_name]['n_estimators'],
                                                      random_state=0)
            model.fit(series['train']['target'])
            # get predictions
            forecasts = model.historical_forecasts(series['test']['target'],
                                                   forecast_horizon=formatter.params['length_pred'],
                                                   stride=1,
                                                   retrain=False,
                                                   verbose=True,
                                                   last_points_only=False)
            forecasts = [scalers['target'].inverse_transform(forecast) for forecast in forecasts]
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(series['test']['target'])

        elif model_name == 'linreg':
            # load and fit model
            model = models.LinearRegressionModel(lags=formatter.params[model_name]['in_len'],
                                                 output_chunk_length=formatter.params['length_pred'])
            model.fit(series['train']['target'])
            # get predictions
            forecasts = model.historical_forecasts(series['test']['target'],
                                                   forecast_horizon=formatter.params['length_pred'], 
                                                   stride=1,
                                                   retrain=False,
                                                   verbose=False,
                                                   last_points_only=False)
            forecasts = [scalers['target'].inverse_transform(forecast) for forecast in forecasts]
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(series['test']['target'])


 
 
 

''' 
#  MODELS: GLUFORMER
device = 'cuda'
for dataset in datasets:
    print(f'Testing {dataset}')

    formatter, series, scalers = load_data(seed=0,
                                           study_file=None,
                                           dataset=dataset,
                                           use_covs=True,
                                           cov_type='dual',
                                           use_static_covs=True)

    # Set model parameters directly
    in_len = 96  # Fixed input length
    label_len = in_len // 3
    out_len = formatter.params['length_pred']
    max_samples_per_ts = 200  # Fixed max samples per time series
    d_model = 512  # Fixed model dimension
    d_fcn = 1024  # Fixed dimension of FCN
    num_enc_layers = 2  # Fixed number of encoder layers
    num_dec_layers = 2  # Fixed number of decoder layers

    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components


    # Create datasets
    dataset_test_glufo = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                covariates=series['test']['future'],
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=True,
                                                array_output_only=True)


  # Build the Gluformer model
    glufo = Gluformer(d_model=d_model,
                      n_heads=12,
                      d_fcn=d_fcn,
                      r_drop=0.2,
                      activ="gelu",
                      num_enc_layers=num_enc_layers,
                      num_dec_layers=num_dec_layers,
                      distil=True,
                      len_seq=in_len,
                      label_len=label_len,
                      len_pred=out_len,
                      num_dynamic_features=num_dynamic_features,
                      num_static_features=num_static_features)

    glufo.to(device)
    glufo.load_state_dict(torch.load(f'./output/tensorboard_gluformer_{dataset}/model.pt', map_location=torch.device(device)))
   

    # get predictions: gluformer
    print('Gluformer')
    forecasts, _ = glufo.predict(dataset_test_glufo,
                                 batch_size=8,
                                 num_samples=10,
                                 device=device,
                                 use_tqdm=True)
    
    forecasts = [scalers['target'].inverse_transform(forecast) for forecast in forecasts]
    
    
    trues = [dataset_test_glufo.evalsample(i) for i in range(len(dataset_test_glufo))]
    trues = scalers['target'].inverse_transform(trues)
    inputs = [dataset_test_glufo[i][0] for i in range(len(dataset_test_glufo))]
    inputs = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_
    save_forecasts[f'gluformer_{dataset}'] = forecasts
    save_trues[f'gluformer_{dataset}'] = trues
    save_inputs[f'gluformer_{dataset}'] = inputs



'''

# save forecasts
with gzip.open('./paper_results/data/compressed_forecasts.pkl', 'wb') as file:
    pickle.dump(save_forecasts, file)

# save true values
with gzip.open('./paper_results/data/compressed_trues.pkl', 'wb') as file:
    pickle.dump(save_trues, file)

# save inputs
with gzip.open('./paper_results/data/compressed_inputs.pkl', 'wb') as file:
    pickle.dump(save_inputs, file)

# Load the saved forecasts, trues, and inputs for further analysis or plotting

# define the color gradient
colors = ['#00264c', '#0a2c62', '#14437f', '#1f5a9d', '#2973bb', '#358ad9', '#4d9af4', '#7bb7ff', '#add5ff', '#e6f3ff']
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# set matplotlib theme to seaborn whitegrid
sns.set_theme(style="whitegrid")

# load forecasts
with gzip.open('./paper_results/data/compressed_forecasts.pkl', 'rb') as file:
    save_forecasts = pickle.load(file)
save_forecasts['linreg_livia_mini'] = [item for sublist in save_forecasts['linreg_livia_mini'] for item in sublist]

# load true values
with gzip.open('./paper_results/data/compressed_trues.pkl', 'rb') as file:
    save_trues = pickle.load(file)

# load inputs
with gzip.open('./paper_results/data/compressed_inputs.pkl', 'rb') as file:
    save_inputs = pickle.load(file)

# plot forecasts on livia_mini for all models
models = ['transformer', 'nhits', 'xgboost', 'linreg', 'gluformer']
models = [f'{name}_livia_mini' for name in models]
offsets = {
    'gluformer_livia_mini': 0, 
    'nhits_livia_mini': 144-48, 
    'xgboost_livia_mini': 144-32,
    'linreg_livia_mini': 144-84, 
    'transformer_livia_mini': 144-96
}
sidx = [50, 75, 100, 150, 175]

fig, axs = plt.subplots(len(models), 5, figsize=(20, 20))


for i, model in enumerate(models):
    forecasts = save_forecasts[model]
    if model not in ['gluformer_livia_mini']:
        forecasts = np.array([forecasts[i].all_values() for i in range(len(forecasts))])
    if 'gluformer' in model:
        # generate samples from predictive distribution
        samples = np.random.normal(loc=forecasts[..., None],
                                   scale=1,
                                   size=(forecasts.shape[0], 
                                         forecasts.shape[1], 
                                         forecasts.shape[2],
                                         30))
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    for j in range(5):
        # put vertical line at 0
        axs[i, j].axvline(x=0, color='black', linestyle='--')
        
        # plot forecasts

        if 'linreg' in model:
            forecast = forecasts[sidx[j] + offsets[model]]
            axs[i, j].plot(np.arange(12), forecast[:, 0], color='red', marker='o')
        if 'transformer' in model:
            forecast = forecasts[sidx[j] + offsets[model]]
            axs[i, j].plot(np.arange(12), forecast[:, 0], color='red', marker='o')
        if 'nhits' in model:
            forecast = forecasts[:, sidx[j] + offsets[model], :, 0]
            median = np.quantile(forecast, 0.5, axis=0)
            axs[i, j].plot(np.arange(12), median, color='red', marker='o')
        if 'gluformer' in model:
            ind = sidx[j] + offsets[model]
            # plot predictive distribution
            for point in range(samples.shape[1]):
                kde = stats.gaussian_kde(samples[ind, point, :])
                maxi, mini = 1.2 * np.max(samples[ind, point, :]), 0.8 * np.min(samples[ind, point, :])
                y_grid = np.linspace(mini, maxi, 200)
                x = kde(y_grid)
                axs[i, j].fill_betweenx(y_grid, x1=point, x2=point - x * 15, 
                                        alpha=0.7, 
                                        edgecolor='black',
                                        color=cmap(point / samples.shape[1]))
            # plot median
            forecast = samples[ind, :, :]
            median = np.quantile(forecast, 0.5, axis=-1)
            axs[i, j].plot(np.arange(12), median, color='red', marker='o')
        # for last row only, xlabel = Time (in 5 minute intervals)
        if i == len(models) - 1:
            axs[i, j].set_xlabel('Time (in 5 minute intervals)')
        # for first column only, ylabel = model name in upper case letters \n Glucose (mg/dL)
        if j == 0:
            axs[i, j].set_ylabel(model.split('_')[0].upper() + '\nGlucose (mg/dL)')

for ax in axs.flatten():
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_fontsize(16)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(16)
    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(20)

# save figure
plt.tight_layout()
plt.savefig('output/plots/figure6.pdf', dpi=300, bbox_inches='tight')