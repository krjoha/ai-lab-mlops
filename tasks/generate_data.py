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

    # Sample data to predict (simulates new incoming data)
    new_data = [
        "this product is absolutely fantastic",
        "really poor quality, very disappointed",
        "good value for money, happy with purchase",
        "disappointing results, not worth it",
        "amazing service and fast delivery",
        "terrible customer support experience",
        "excellent quality, highly recommend",
        "worst purchase ever made",
        "great experience overall",
        "completely unsatisfied with this"
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'text': new_data,
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