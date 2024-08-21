import typer
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = typer.Typer()

def load_model(model_path: Path) -> nn.Module:
    """Load a PyTorch model from a given path."""
    model = torch.load(model_path)
    model.eval()
    return model

def prepare_input_data(data: pd.DataFrame) -> torch.Tensor:
    """
    Prepare input data for the transformer model.
    This function should be adapted to match the specific input requirements of your model.
    """
    # Example: Assuming the data needs to be of shape [batch_size, sequence_length, num_features]
    # This example assumes that your dataframe is already in the appropriate sequence order.
    # Adjust this based on your actual model's input needs.
    input_tensor = torch.tensor(data.values, dtype=torch.float32)

    # If required, reshape or manipulate `input_tensor` to match your model's input shape.
    # e.g., input_tensor = input_tensor.view(batch_size, sequence_length, num_features)

    return input_tensor

@app.command()
def process_csv(
        csv_path: Path = typer.Option("./raw_data/anton.csv", help="Path to the CSV file with glucose plots."),
        model_path: Path = typer.Option(
            "output/models/anton/gluformer_1samples_400epochs_10heads_32batch_geluactivation_anton.pth",
            help="Path to the PyTorch model."
        ),
        output_path: Path = typer.Option("output/anton", help="Path where the predictions will be saved."),
        ground_truth: bool = typer.Option(False, help="Flag to include ground truth in predictions.")
):
    """Process a CSV file with glucose plots, generate predictions, and optionally compare with ground truth."""

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Load the PyTorch model
    model = load_model(model_path)

    if ground_truth:
        # Split the data into training and ground truth (last 12 values for ground truth)
        input_data = df.iloc[:-12]
        ground_truth_data = df.iloc[-12:]

        # Prepare input data for model prediction
        input_tensor = prepare_input_data(input_data)

        # Generate predictions for the input data
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = predictions.numpy().flatten()  # Ensure predictions are flattened

        # Prepare ground truth data for model prediction
        ground_truth_tensor = prepare_input_data(ground_truth_data)

        # Generate predictions for the ground truth data
        with torch.no_grad():
            ground_truth_predictions = model(ground_truth_tensor)
            ground_truth_predictions = ground_truth_predictions.numpy().flatten()

        # Calculate MAE and RMSE
        mae = mean_absolute_error(ground_truth_data['target_column_name'], ground_truth_predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth_data['target_column_name'], ground_truth_predictions))

        typer.echo(f"MAE: {mae}")
        typer.echo(f"RMSE: {rmse}")

        # Save metrics to a metrics.csv file in the same directory as output_path
        metrics_path = output_path.parent / "metrics.csv"
        metrics_df = pd.DataFrame({"MAE": [mae], "RMSE": [rmse]})
        metrics_df.to_csv(metrics_path, index=False)
        typer.echo(f"Metrics saved to {metrics_path}")

        # Append predictions to the dataframe
        df['predictions'] = np.append(predictions, ground_truth_predictions)
    else:
        # Prepare input data for model prediction
        input_tensor = prepare_input_data(df)

        # Generate predictions for the entire dataset
        with torch.no_grad():
            predictions = model(input_tensor)
            df['predictions'] = predictions.numpy().flatten()

    # Save the results to the output path
    df.to_csv(output_path, index=False)
    typer.echo(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    app()
