import sys
import os
import yaml
import datetime
import argparse
from functools import partial

import torch
import optuna
from torch.utils.tensorboard import SummaryWriter

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from lib.gluformer.model import Gluformer
from lib.gluformer.utils.evaluation import test
from utils.darts_training import print_callback
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual

# define objective function
def objective(trial):
    # set parameters
    out_len = formatter.params['length_pred']
    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components
    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 
                           f'../output/tensorboard_gluformer_{args.dataset}/run{trial.number}'))
    model_path = os.path.join(os.path.dirname(__file__),
                              f'../output/tensorboard_gluformer_{args.dataset}/model.pt')
    # suggest hyperparameters: input size
    in_len = trial.suggest_int("in_len", 96, formatter.params['max_length_input'], step=12)
    label_len = in_len // 3
    max_samples_per_ts = trial.suggest_int("max_samples_per_ts", 50, 200, step=50)
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    d_model = trial.suggest_int("d_model", 256, 512, step=128)
    n_heads = trial.suggest_int("n_heads", 4, 12, step=4)
    d_fcn = trial.suggest_int("d_fcn", 512, 2048, step=128)  
    num_enc_layers = trial.suggest_int("num_enc_layers", 1, 4, step=1)
    num_dec_layers = trial.suggest_int("num_dec_layers", 1, 4, step=1)

    # create datasets
    dataset_train = SamplingDatasetDual(series['train']['target'],
                                        series['train']['future'],
                                        output_chunk_length=out_len,
                                        input_chunk_length=in_len,
                                        use_static_covariates=True,
                                        max_samples_per_ts=max_samples_per_ts,
                                        )
    dataset_val = SamplingDatasetDual(series['val']['target'],
                                      series['val']['future'],   
                                      output_chunk_length=out_len,
                                      input_chunk_length=in_len,
                                      use_static_covariates=True,)
    
    # build the NHiTSModel model
    model = Gluformer(d_model = d_model, 
                      n_heads = n_heads, 
                      d_fcn = d_fcn, 
                      r_drop = 0.2, 
                      activ = 'relu', 
                      num_enc_layers = num_enc_layers, 
                      num_dec_layers = num_dec_layers,
                      distil = True, 
                      len_seq = in_len,
                      label_len = label_len,
                      len_pred = out_len,
                      num_dynamic_features = num_dynamic_features,
                      num_static_features = num_static_features,)

    # train the model
    model.fit(dataset_train,
              dataset_val,
              learning_rate = 1e-4,
              batch_size = 32,
              epochs = 100,
              num_samples = 1,
              device = device,
              model_path = model_path,
              trial = None,
              logger = writer)

    # predict on the validation set
    dataset_val = SamplingDatasetInferenceDual(target_series=series['val']['target'],
                                               covariates=series['val']['future'],
                                               input_chunk_length=in_len,
                                               output_chunk_length=out_len,
                                               use_static_covariates=True,
                                               array_output_only=True)
    predictions, logvar = model.predict(dataset_val,
                                        batch_size=32,
                                        num_samples=3,
                                        device=device,)
    trues = np.array([dataset_val.evalsample(i).values() for i in range(len(dataset_val))])
    # rescale predictions
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    var = np.exp(logvar) / scalers['target'].scale_**2
    # test
    errors, _, _ = test(trues, predictions, var=var)
    avg_error = np.mean(errors[:, 1])

    return avg_error

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--optuna', type=str, default='True')
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()
reductions = [args.reduction1, args.reduction2, args.reduction3]
if __name__ == '__main__':
    # define device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # load data
    study_file = f'./output/gluformer_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    formatter, series, scalers = load_data(seed=0, 
                                           study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True, 
                                           cov_type='dual',
                                           use_static_covs=True)
    
    # hyperparameter optimization
    best_params = None
    if args.optuna == 'True':
        study = optuna.create_study(direction="minimize")
        print_call = partial(print_callback, study_file=study_file)
        study.optimize(objective, n_trials=50, 
                       callbacks=[print_call], 
                       catch=(np.linalg.LinAlgError, KeyError))
        best_params = study.best_trial.params
    else:
        assert formatter.params["gluformer"] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params["gluformer"]

    # set parameters
    out_len = formatter.params['length_pred']
    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components
    model_path = os.path.join(os.path.dirname(__file__),
                              f'../output/tensorboard_gluformer_{args.dataset}/model.pt')
    # suggest hyperparameters: input size
    in_len = best_params["in_len"]
    label_len = in_len // 3
    max_samples_per_ts = best_params["max_samples_per_ts"]
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    d_model = best_params["d_model"]
    n_heads = best_params["n_heads"]
    d_fcn = best_params["d_fcn"]
    num_enc_layers = best_params["num_enc_layers"]
    num_dec_layers = best_params["num_dec_layers"]

    # Set model seed
    model_seeds = list(range(10, 20))
    id_errors_model = {key: [] for key in reductions if key is not None}
    ood_errors_model = {key: [] for key in reductions if key is not None}
    id_likelihoods_model = []; ood_likelihoods_model = []
    id_cal_errors_model = []; ood_cal_errors_model = []
    for model_seed in model_seeds:
        # Backtest on the test set
        seeds = list(range(1, 3))
        id_errors_cv = {key: [] for key in reductions if key is not None}
        ood_errors_cv = {key: [] for key in reductions if key is not None}
        id_likelihoods_cv = []; ood_likelihoods_cv = []
        id_cal_errors_cv = []; ood_cal_errors_cv = []
        for seed in seeds:
            writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 
                           f'../output/tensorboard_gluformer_{args.dataset}/test_run{model_seed}_{seed}'))
            formatter, series, scalers = reshuffle_data(formatter=formatter, 
                                                        seed=seed, 
                                                        use_covs=True,
                                                        cov_type='past',
                                                        use_static_covs=True)
            # create datasets
            dataset_train = SamplingDatasetDual(series['train']['target'],
                                                series['train']['future'],
                                                output_chunk_length=out_len,
                                                input_chunk_length=in_len,
                                                use_static_covariates=True,
                                                max_samples_per_ts=max_samples_per_ts,)
            dataset_val = SamplingDatasetDual(series['val']['target'],
                                              series['val']['future'],   
                                              output_chunk_length=out_len,
                                              input_chunk_length=in_len,
                                              use_static_covariates=True,)
            dataset_test = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                        covariates=series['test']['future'],
                                                        input_chunk_length=in_len,
                                                        output_chunk_length=out_len,
                                                        use_static_covariates=True,
                                                        array_output_only=True)
            dataset_test_ood = SamplingDatasetInferenceDual(target_series=series['test_ood']['target'],
                                                            covariates=series['test_ood']['future'],
                                                            input_chunk_length=in_len,
                                                            output_chunk_length=out_len,
                                                            use_static_covariates=True,
                                                            array_output_only=True)
            # build the Gluformer model
            model = Gluformer(d_model = d_model, 
                            n_heads = n_heads, 
                            d_fcn = d_fcn, 
                            r_drop = 0.2, 
                            activ = 'relu', 
                            num_enc_layers = num_enc_layers, 
                            num_dec_layers = num_dec_layers,
                            distil = True, 
                            len_seq = in_len,
                            label_len = label_len,
                            len_pred = out_len,
                            num_dynamic_features = num_dynamic_features,
                            num_static_features = num_static_features,)

            # train the model
            model.fit(dataset_train,
                      dataset_val,
                      learning_rate = 1e-4,
                      batch_size = 32,
                      epochs = 100,
                      num_samples = 1,
                      device = device,
                      model_path = model_path,
                      trial = None,
                      logger = writer)

            # backtest on the test set
            predictions, logvar = model.predict(dataset_test,
                                                batch_size=32,
                                                num_samples=3,
                                                device=device,)
            trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])
            trues = (trues - scalers['target'].min_) / scalers['target'].scale_
            predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
            var = np.exp(logvar) / scalers['target'].scale_**2
            id_errors_sample, id_likelihood_sample, id_cal_errors_sample = test(trues, predictions, var=var)

            # backtest on the ood test set
            predictions, logvar = model.predict(dataset_test_ood,
                                                batch_size=32,
                                                num_samples=3,
                                                device=device,)
            trues = np.array([dataset_test_ood.evalsample(i).values() for i in range(len(dataset_test_ood))])
            trues = (trues - scalers['target'].min_) / scalers['target'].scale_
            predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
            var = np.exp(logvar) / scalers['target'].scale_**2
            ood_errors_sample, ood_likelihood_sample, ood_cal_errors_sample = test(trues, predictions, var=var)
            
            # compute, save, and print results
            with open(study_file, "a") as f:
                for reduction in reductions:  
                    if reduction is not None:
                        # compute
                        reduction_f = getattr(np, reduction)
                        id_errors_sample_red = reduction_f(id_errors_sample, axis=0)
                        ood_errors_sample_red = reduction_f(ood_errors_sample, axis=0)
                        # save
                        id_errors_cv[reduction].append(id_errors_sample_red)
                        ood_errors_cv[reduction].append(ood_errors_sample_red)
                        # print
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID {reduction} of (MSE, MAE): {id_errors_sample_red}\n")
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD {reduction} of (MSE, MAE) stats: {ood_errors_sample_red}\n")
                # save
                id_likelihoods_cv.append(id_likelihood_sample)
                ood_likelihoods_cv.append(ood_likelihood_sample)
                id_cal_errors_cv.append(id_cal_errors_sample)
                ood_cal_errors_cv.append(ood_cal_errors_sample)
                # print
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID likelihoods: {id_likelihood_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD likelihoods: {ood_likelihood_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID calibration errors: {id_cal_errors_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD calibration errors: {ood_cal_errors_sample}\n")

        # compute, save, and print results
        with open(study_file, "a") as f:
            for reduction in reductions:
                if reduction is not None:
                    # compute
                    id_errors_cv[reduction] = np.vstack(id_errors_cv[reduction])
                    ood_errors_cv[reduction] = np.vstack(ood_errors_cv[reduction])
                    id_errors_cv[reduction] = np.mean(id_errors_cv[reduction], axis=0)
                    ood_errors_cv[reduction] = np.mean(ood_errors_cv[reduction], axis=0)
                    # save
                    id_errors_model[reduction].append(id_errors_cv[reduction])
                    ood_errors_model[reduction].append(ood_errors_cv[reduction])
                    # print
                    f.write(f"\tModel Seed: {model_seed} ID {reduction} of (MSE, MAE): {id_errors_cv[reduction]}\n")
                    f.write(f"\tModel Seed: {model_seed} OOD {reduction} of (MSE, MAE): {ood_errors_cv[reduction]}\n")
            # compute
            id_likelihoods_cv = np.mean(id_likelihoods_cv)
            ood_likelihoods_cv = np.mean(ood_likelihoods_cv)
            id_cal_errors_cv = np.vstack(id_cal_errors_cv)
            ood_cal_errors_cv = np.vstack(ood_cal_errors_cv)
            id_cal_errors_cv = np.mean(id_cal_errors_cv, axis=0)
            ood_cal_errors_cv = np.mean(ood_cal_errors_cv, axis=0)
            # save
            id_likelihoods_model.append(id_likelihoods_cv)
            ood_likelihoods_model.append(ood_likelihoods_cv)
            id_cal_errors_model.append(id_cal_errors_cv)
            ood_cal_errors_model.append(ood_cal_errors_cv)
            # print
            f.write(f"\tModel Seed: {model_seed} ID likelihoods: {id_likelihoods_cv}\n")
            f.write(f"\tModel Seed: {model_seed} OOD likelihoods: {ood_likelihoods_cv}\n")
            f.write(f"\tModel Seed: {model_seed} ID calibration errors: {id_cal_errors_cv}\n")
            f.write(f"\tModel Seed: {model_seed} OOD calibration errors: {ood_cal_errors_cv}\n")
                
    # compute, save, and print results
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # compute mean and std
                id_errors_model[reduction] = np.vstack(id_errors_model[reduction])
                ood_errors_model[reduction] = np.vstack(ood_errors_model[reduction])
                id_mean = np.mean(id_errors_model[reduction], axis=0)
                ood_mean = np.mean(ood_errors_model[reduction], axis=0)
                id_std = np.std(id_errors_model[reduction], axis=0)
                ood_std = np.std(ood_errors_model[reduction], axis=0)
                # print
                f.write(f"ID {reduction} of (MSE, MAE): {id_mean.tolist()} +- {id_std.tolist()}\n")
                f.write(f"OOD {reduction} of (MSE, MAE): {ood_mean.tolist()} +- {ood_std.tolist()}\n")
        # compute mean and std of likelihoods
        id_mean = np.mean(id_likelihoods_model)
        ood_mean = np.mean(ood_likelihoods_model)
        id_std = np.std(id_likelihoods_model)
        ood_std = np.std(ood_likelihoods_model)
        # print
        f.write(f"ID likelihoods: {id_mean} +- {id_std}\n")
        f.write(f"OOD likelihoods: {ood_mean} +- {ood_std}\n")
        # compute mean and std of calibration errors
        id_cal_errors_model = np.vstack(id_cal_errors_model)
        ood_cal_errors_model = np.vstack(ood_cal_errors_model)
        id_means = np.mean(id_cal_errors_model, axis=0)
        ood_means = np.mean(ood_cal_errors_model, axis=0)
        id_stds = np.std(id_cal_errors_model, axis=0)
        ood_stds = np.std(ood_cal_errors_model, axis=0)
        # print
        f.write(f"ID calibration errors: {id_means.tolist()} +- {id_stds.tolist()}\n")
        f.write(f"OOD calibration errors: {ood_means.tolist()} +- {ood_stds.tolist()}\n")

