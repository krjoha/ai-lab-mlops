#!/usr/bin/env python3
"""
Data generation task for MLOps pipeline.
Simulates data ingestion - in production this would fetch from APIs, databases, etc.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def generate_data():
    """Generate sample data for batch prediction"""
    print("Starting data generation task...")

    # Create output directory
    data_dir = Path("./artifacts/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load batch prediction samples and sample deterministically from them
    samples_path = data_dir / "batch_prediction_samples.csv"
    if not samples_path.exists():
        raise FileNotFoundError(f"Batch prediction samples not found at {samples_path}. Please ensure the sample CSV file exists.")

    print(f"Loading samples from {samples_path}")
    samples_df = pd.read_csv(samples_path)

    # Sample randomly (different results each run for varied Airflow outputs)
    # Take a subset of the samples to simulate new incoming data
    import numpy as np
    sample_size = min(20, len(samples_df))  # Sample exactly 20 items or all if fewer
    sampled_indices = np.random.choice(len(samples_df), size=sample_size, replace=False)
    sampled_data = samples_df.iloc[sampled_indices]

    # Create DataFrame with additional metadata
    df = pd.DataFrame({
        'text': sampled_data['text'].values,
        'created_at': datetime.now().isoformat(),
        'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    })

    print(f"Generated {len(df)} samples for prediction")

    # Save data
    data_path = data_dir / "input_data.csv"
    df.to_csv(data_path, index=False)

    # Save metadata
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "sample_count": len(df),
        "batch_id": df['batch_id'].iloc[0],
        "data_path": str(data_path)
    }

    metadata_path = data_dir / "data_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Data saved to {data_path}")
    print(f"Metadata saved to {metadata_path}")
    print("Data generation task completed successfully")

if __name__ == "__main__":
    generate_data()