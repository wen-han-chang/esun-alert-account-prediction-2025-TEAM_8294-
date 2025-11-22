"""Project entry point.

This module runs the full pipeline:
1) feature engineering to generate training / prediction features
2) model training and prediction to produce the final submission file.
"""
from Preprocess.feature_engineering_timefix import main as preprocess_main
from Model.model import main as model_main

if __name__ == "__main__":
    preprocess_main()
    model_main()
