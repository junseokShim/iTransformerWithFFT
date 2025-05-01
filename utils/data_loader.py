# data_loader.py
import pandas as pd
import glob
import os


def load_parquet_files(data_dir):
    parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))
    data = {}
    for file_path in parquet_files:
        name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
        data[name] = pd.read_parquet(file_path)
    return data


def load_csv_files(metrics_path, submission_path):
    metrics_train = pd.read_csv(metrics_path)
    sample_submission = pd.read_csv(submission_path)
    return metrics_train, sample_submission


def load_specific_parquet_files(data_dir, filenames):
    data = {}
    for name in filenames:
        file_path = os.path.join(data_dir, f'ch2025_{name}.parquet')
        data[name] = pd.read_parquet(file_path)
    return data


def load_all_data(data_dir, metrics_path, submission_path, filenames):
    parquet_data = load_specific_parquet_files(data_dir, filenames)
    metrics_train, sample_submission = load_csv_files(metrics_path, submission_path)
    return parquet_data, metrics_train, sample_submission