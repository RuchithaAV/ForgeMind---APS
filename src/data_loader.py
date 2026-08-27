from pathlib import Path

import pandas as pd

def load_aps_data(csv_path):

    """
    Load an APS CSV file and separate predictors and target.

    Parameters
    ----------
    csv_path:
        Location of the APS CSV file.

    Returns
    -------
    X:
        Predictor features.

    y:
        Binary target where:
        0 = non-APS failure
        1 = APS failure
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        sep=',',
        na_values='na'
    )

    if "class" not in df.columns:
        raise ValueError("Column 'class' not found in the dataframe!")

    valid_labels = {"neg", "pos"}

    actual_labels = set(
        df["class"].dropna().unique()
    )

    if not actual_labels == valid_labels:
        raise ValueError(f"Invalid class labels found. Expected {valid_labels} but got {actual_labels}.")
        
    X = df.drop(columns="class")

    y = df["class"].map({"neg":0, "pos":1})

    return X, y