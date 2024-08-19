import sys
import os
import typer
from pathlib import Path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# Improve library imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.gluformer.model import Gluformer
from lib.gluformer.utils.evaluation import test
from utils.darts_processing import load_data
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual

def main(
        dataset: str = 'weinstock',
        gpu_id: int = 0,
        output_dir: Path = Path('./output/gluformer')
):
    # Define device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Prepare paths
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{dataset}_gluformer_model.pt"
    metrics_path = output_dir / f"{dataset}_gluformer_metrics.csv"

    # Load data
    formatter, series, scalers = load_data(seed=0, dataset=dataset, use_covs=True, cov_type='dual', use_static_covs=True)

    # Set parameters
    out_len = formatter.params['length_pred']
    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components

    in_len = formatter.params['max_length_input']  # use the max input length
    label_len = in_len // 3
    max_samples_per_ts = None  # unlimited samples

    # Model hyperparameters (for simplicity, using fixed values)
    d_model = 256
    n_heads = 8
    d_fcn = 1024
    num_enc_layers = 2
    num_dec_layers = 2

    # Create datasets
    dataset_train = SamplingDatasetDual(series['train']['target'], series['train']['future'], output_chunk_length=out_len, input_chunk_length=in_len, use_static_covariates=True, max_samples_per_ts=max_samples_per_ts)
    dataset_test = SamplingDatasetInferenceDual(target_series=series['test']['target'], covariates=series['test']['future'], input_chunk_length=in_len, output_chunk_length=out_len, use_static_covariates=True, array_output_only=True)

    # Build the Gluformer model
    model = Gluformer(d_model=d_model, n_heads=n_heads, d_fcn=d_fcn, r_drop=0.2, activ='relu', num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers, distil=True, len_seq=in_len, label_len=label_len, len_pred=out_len, num_dynamic_features=num_dynamic_features, num_static_features=num_static_features)

    # Train the model
    writer = SummaryWriter(str(output_dir / 'tensorboard'))
    model.fit(dataset_train, dataset_test, learning_rate=1e-4, batch_size=32, epochs=100, num_samples=1, device=device, model_path=str(model_path), trial=None, logger=writer)

    # Test the model
    predictions, logvar = model.predict(dataset_test, batch_size=32, num_samples=3, device=device)
    trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])

    # Calculate MAE and RMSE (No scaling)
    mae = np.mean(np.abs(trues - predictions))
    rmse = np.sqrt(np.mean((trues - predictions) ** 2))

    # Save the metrics
    np.savetxt(metrics_path, np.array([['MAE', mae], ['RMSE', rmse]]), delimiter=',', fmt='%s')

    print(f"Training complete. Model saved to {model_path}. Metrics saved to {metrics_path}.")

if __name__ == "__main__":
    # python lib/gluformer_simple.py --dataset weinstock --output-dir simple
    typer.run(main)
