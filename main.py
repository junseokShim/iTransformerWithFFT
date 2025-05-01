import numpy as np
import pandas as pd
import glob
import random
import os
import seaborn as sns
import ast

from utils.preprocessing import *

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from models.itransformer import *
from train import *
from prediction import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train iTransformer model and generate submission file.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--submission_name', type=str, default='submission_final_fft_256_itransformer.csv',
                        help='Filename for final submission output')
    return parser.parse_args()

# -------------------------- 공통 설정 --------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_parquet_files(data_dir):
    parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))
    lifelog_data = {}
    for file_path in parquet_files:
        name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
        lifelog_data[name] = pd.read_parquet(file_path)
        print(f"✅ Loaded: {name}, shape = {lifelog_data[name].shape}")
    return lifelog_data

def assign_to_globals(data_dict):
    for key, df in data_dict.items():
        globals()[f"{key}_df"] = df

# -------------------------- 데이터 분리 --------------------------
def split_test_train(df, subject_col='subject_id', timestamp_col='timestamp', test_keys=None):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df['date_only'] = df[timestamp_col].dt.date
    df['key'] = list(zip(df[subject_col], df['date_only']))

    test_df = df[df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    train_df = df[~df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    return test_df, train_df

def split_all_dataframes(dataframes, test_keys):
    result = {}
    for name, (df, ts_col) in dataframes.items():
        print(f"⏳ {name} 분리 중...")
        test_df, train_df = split_test_train(df.copy(), subject_col='subject_id', timestamp_col=ts_col, test_keys=test_keys)
        result[f"{name}_test"] = test_df
        result[f"{name}_train"] = train_df
        print(f"✅ {name}_test → {test_df.shape}, {name}_train → {train_df.shape}")
    return result

# -------------------------- 전처리 --------------------------
def sanitize_column_names(df):
    df.columns = (
        df.columns
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df

# -------------------------- 학습용 데이터 생성 --------------------------
def prepare_train_test_data(metrics_train, merged_df):
    metrics_train['lifelog_date'] = pd.to_datetime(metrics_train['lifelog_date']).dt.date
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date

    metrics_train = metrics_train.rename(columns={'lifelog_date': 'date'})

    train_df = pd.merge(metrics_train, merged_df, on=['subject_id', 'date'], how='inner')

    merged_keys = merged_df[['subject_id', 'date']]
    train_keys = metrics_train[['subject_id', 'date']]
    test_keys = pd.merge(merged_keys, train_keys, on=['subject_id', 'date'], how='left', indicator=True)
    test_keys = test_keys[test_keys['_merge'] == 'left_only'].drop(columns=['_merge'])

    test_df = pd.merge(test_keys, merged_df, on=['subject_id', 'date'], how='left')
    return train_df, test_df

# -------------------------- 모델 학습 --------------------------
def train_and_predict_models(X_tensor, test_X_tensor, train_df, targets_binary, target_multiclass, epochs, lr, batch_size):
    binary_preds = {}
    for col in targets_binary:
        y_tensor = torch.tensor(train_df[col].values, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model_bin = TransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=2)
        train_model(model_bin, dataloader, nn.CrossEntropyLoss(), optim.Adam(model_bin.parameters(), lr=lr), epochs=epochs)
        binary_preds[col] = predict(model_bin, test_X_tensor)

    y_multi_tensor = torch.tensor(train_df[target_multiclass].values, dtype=torch.long)
    dataset_multi = TensorDataset(X_tensor, y_multi_tensor)
    dataloader_multi = DataLoader(dataset_multi, batch_size=batch_size, shuffle=True)

    model_multi = TransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=3)
    train_model(model_multi, dataloader_multi, nn.CrossEntropyLoss(), optim.Adam(model_multi.parameters(), lr=lr), epochs=epochs)
    multiclass_pred = predict(model_multi, test_X_tensor)

    return binary_preds, multiclass_pred


# -------------------------- 제출 파일 생성 --------------------------
def generate_submission(sample_submission, binary_preds, multiclass_pred, filename):
    sample_submission['lifelog_date'] = pd.to_datetime(sample_submission['lifelog_date']).dt.date
    submission_final = sample_submission[['subject_id', 'sleep_date', 'lifelog_date']].copy()
    submission_final['ID'] = submission_final['subject_id'] + '_' + submission_final['lifelog_date'].astype(str)

    submission_final['S1'] = multiclass_pred
    for col in binary_preds:
        submission_final[col] = binary_preds[col].astype(int)

    submission_final = submission_final[['subject_id', 'sleep_date', 'lifelog_date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']]
    submission_final.to_csv(filename, index=False)

# -------------------------- 메인 --------------------------
def main():
    seed_everything(42)
    args = parse_args()
    data_dir = 'dataset/ch2025_data_items'

    lifelog_data = load_parquet_files(data_dir)
    assign_to_globals(lifelog_data)

    metrics_train = pd.read_csv('dataset/ch2025_data_items/ch2025_metrics_train.csv')
    sample_submission = pd.read_csv('dataset/ch2025_submission_sample.csv')
    merged_df = pd.read_csv('dataset/merged_df.csv')

    test_keys = set(zip(pd.to_datetime(sample_submission['lifelog_date']).dt.date, sample_submission['subject_id']))

    mACStatus_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mACStatus.parquet')
    mActivity_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mActivity.parquet')
    mAmbience_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mAmbience.parquet')
    mBle_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mBle.parquet')
    mGps_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mGps.parquet')
    mLight_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mLight.parquet')
    mScreenStatus_df =  pd.read_parquet('dataset/ch2025_data_items/ch2025_mScreenStatus.parquet')
    mUsageStats_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mUsageStats.parquet')
    mWifi_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_mWifi.parquet')
    wHr_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_wHr.parquet')
    wLight_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_wLight.parquet')
    wPedo_df = pd.read_parquet('dataset/ch2025_data_items/ch2025_wPedo.parquet')

    dataframes = {
        'mACStatus': (mACStatus_df, 'timestamp'),
        'mActivity': (mActivity_df, 'timestamp'),
        'mAmbience': (mAmbience_df, 'timestamp'),
        'mBle': (mBle_df, 'timestamp'),
        'mGps': (mGps_df, 'timestamp'),
        'mLight': (mLight_df, 'timestamp'),
        'mScreenStatus': (mScreenStatus_df, 'timestamp'),
        'mUsageStats': (mUsageStats_df, 'timestamp'),
        'mWifi': (mWifi_df, 'timestamp'),
        'wHr': (wHr_df, 'timestamp'),
        'wLight': (wLight_df, 'timestamp'),
        'wPedo': (wPedo_df, 'timestamp'),
    }

    split_results = split_all_dataframes(dataframes, test_keys)
    train_df, test_df = prepare_train_test_data(metrics_train, merged_df)

    targets_binary = ['Q1', 'Q2', 'Q3', 'S2', 'S3']
    target_multiclass = 'S1'

    X = train_df.drop(columns=['subject_id', 'sleep_date', 'date'] + targets_binary + [target_multiclass]).fillna(0)
    test_X = test_df.drop(columns=['subject_id', 'date']).fillna(0)

    X = sanitize_column_names(X)
    test_X = sanitize_column_names(test_X)

    # Training and test data preparation
    X_tensor, test_X_tensor = prepare_data_itransformer(X, test_X)
    binary_preds, multiclass_pred = train_and_predict_models(X_tensor, test_X_tensor, train_df, targets_binary, target_multiclass, args.epochs, args.lr, args.batch_size)
    generate_submission(sample_submission, binary_preds, multiclass_pred, args.submission_name)

if __name__ == "__main__":
    main()
