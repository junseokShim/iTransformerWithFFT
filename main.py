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
from models.iftransformer import *
from models.transformer import *
from models.advanced_transformer import *
from models.fftformer import * 

from train import *
from prediction import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train iTransformer model and generate submission file.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=float, default=64, help='Hidden layer dimension size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--model_name', type=str, default='itransformer', help='Model name')
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
def train_and_predict_models(X_tensor, test_X_tensor, train_df, targets_binary, target_multiclass, epochs, lr, batch_size, hidden_dim, model_name, num_heads):
    binary_preds = {}
    binary_loss = {}
    binary_f1 = {}

    print(X_tensor.shape, test_X_tensor.shape)

    f1_scores = []

    # 모델 학습 및 예측 (pretrain)
    if model_name == 'pretrain':
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = ITransformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_pretraining_model_with_val(model, dataloader, optimizer, epochs=epochs)
        torch.save(model.state_dict(), f"weights/pretrained_itransformer_{int(hidden_dim)}_{int(num_heads)}.pt")
        return

    elif model_name == 'pretrain_fftformer':
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = FFTformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_pretraining_model_with_val(model, dataloader, optimizer, epochs=epochs)
        torch.save(model.state_dict(), f"weights/pretrained_fftformer_{int(hidden_dim)}_{int(num_heads)}.pt")
        return

    for col in targets_binary:
        y_tensor = torch.tensor(train_df[col].values, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if model_name == 'itransformer':
            model_bin = ITransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=2, hidden_dim=int(hidden_dim), num_heads=int(num_heads)) #SOTA

        elif model_name == 'fine-tunning':
            # 1. Pretraining된 모델 객체 생성 및 weight 불러오기
            pretrained_model = ITransformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
            pretrained_model.load_state_dict(torch.load(f"weights/pretrained_itransformer_{int(hidden_dim)}_{int(num_heads)}.pt"))
            
            # 2. encoder만 추출
            pretrained_encoder = pretrained_model.encoder

            # ✅ FourierSelfAttention 파라미터 freeze
            for param in pretrained_encoder.parameters():
                param.requires_grad = False
                        
            model_bin = ITransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=2, hidden_dim=int(hidden_dim), num_heads=int(num_heads), pretrained_encoder=pretrained_encoder)

        elif model_name == 'fine-tunning_fftformer':
            # 1. Pretraining된 모델 객체 생성 및 weight 불러오기
            pretrained_model = FFTformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
            pretrained_model.load_state_dict(torch.load(f"weights/pretrained_fftformer_{int(hidden_dim)}_{int(num_heads)}.pt"))

            
            # 2. encoder만 추출
            pretrained_encoder = pretrained_model.encoder_layers

            # ✅ FourierSelfAttention 파라미터 freeze
            for param in pretrained_encoder.parameters():
                param.requires_grad = False
                        
            model_bin = FFTformerClassifier(input_dim=X_tensor.shape[-1], num_classes=2, hidden_dim=int(hidden_dim), num_heads=int(num_heads), pretrained_encoder=pretrained_encoder)

        binary_f1[col] = train_model(model_bin, dataloader, nn.CrossEntropyLoss(), optim.Adam(model_bin.parameters(), lr=lr), col = col, epochs=epochs)
        binary_preds[col] = predict(model_bin, test_X_tensor, col)
        f1_scores.append(binary_f1[col])

    y_multi_tensor = torch.tensor(train_df[target_multiclass].values, dtype=torch.long)
    dataset_multi = TensorDataset(X_tensor, y_multi_tensor)
    dataloader_multi = DataLoader(dataset_multi, batch_size=batch_size, shuffle=True)

    if model_name == 'itransformer':
        model_multi = ITransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=3, hidden_dim=int(hidden_dim), num_heads=int(num_heads)) # SOTA

    elif model_name == 'fine-tunning':
        pretrained_model = ITransformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
        pretrained_model.load_state_dict(torch.load(f"weights/pretrained_itransformer_{int(hidden_dim)}_{int(num_heads)}.pt"))

        # 2. encoder만 추출
        pretrained_encoder = pretrained_model.encoder

        # ✅ FourierSelfAttention 파라미터 freeze
        for param in pretrained_encoder.parameters():
                param.requires_grad = False
        model_multi = ITransformerClassifier(input_dim=X_tensor.shape[-1], num_classes=3, hidden_dim=int(hidden_dim), num_heads=int(num_heads), pretrained_encoder=pretrained_encoder)

    elif model_name == 'fine-tunning_fftformer':
        pretrained_model = FFTformerPretrainer(input_dim=X_tensor.shape[-1], hidden_dim=int(hidden_dim), num_heads=int(num_heads))
        pretrained_model.load_state_dict(torch.load(f"weights/pretrained_fftformer_{int(hidden_dim)}_{int(num_heads)}.pt"))

        # 2. encoder만 추출
        pretrained_encoder = pretrained_model.encoder_layers

        # ✅ FourierSelfAttention 파라미터 freeze
        for param in pretrained_encoder.parameters():
                param.requires_grad = False
        model_multi = FFTformerClassifier(input_dim=X_tensor.shape[-1], num_classes=3, hidden_dim=int(hidden_dim), num_heads=int(num_heads), pretrained_encoder=pretrained_encoder)


    multiclass_f1 = train_model(model_multi, dataloader_multi, nn.CrossEntropyLoss(), optim.Adam(model_multi.parameters(), lr=lr), col = 'S1', epochs=epochs)
    f1_scores.append(multiclass_f1)

    multiclass_pred = predict(model_multi, test_X_tensor, 'S1')

    avg_f1 = sum(f1_scores) / len(f1_scores)    

    with open('logs.txt', 'a') as f:
        f.write(f"hidden layer dim_size: {hidden_dim}\n")
        f.write(f"multi head num : {num_heads}\n")
        f.write(f"이진 분류 Valid Score : {[f'{col}: {binary_f1[col]}' for col in targets_binary]}\n")
        f.write(f"다중 클래스 Valid Score : {'S1'}, {multiclass_f1}\n")
        f.write(f"Avg F1 Score : {avg_f1}\n")
        f.write("\n")
    
    return binary_preds, multiclass_pred, avg_f1


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
    print(f"✅ 제출 파일 생성 완료: {filename}")

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

    train_df, test_df = prepare_train_test_data(metrics_train, merged_df)

    targets_binary = ['Q1', 'Q2', 'Q3', 'S2', 'S3']
    target_multiclass = 'S1'

    X = train_df.drop(columns=['subject_id', 'sleep_date', 'date'] + targets_binary + [target_multiclass]).fillna(0)
    test_X = test_df.drop(columns=['subject_id', 'date']).fillna(0)

    X = sanitize_column_names(X)
    test_X = sanitize_column_names(test_X)

    # Training and test data preparation
    X_tensor, test_X_tensor = prepare_data_itransformer(X, test_X)

    if args.model_name == 'pretrain' or args.model_name == 'pretrain_fftformer':
        train_and_predict_models(X_tensor, test_X_tensor, train_df, \
                                targets_binary, target_multiclass, \
                                args.epochs, args.lr, args.batch_size, args.hidden, args.model_name, args.num_heads)
        return
    
    binary_preds, multiclass_pred, avg_f1_score = train_and_predict_models(X_tensor, test_X_tensor, train_df, \
                                    targets_binary, target_multiclass, \
                                    args.epochs, args.lr, args.batch_size, args.hidden, args.model_name, args.num_heads)
    
    generate_submission(sample_submission, binary_preds, multiclass_pred, f'submission/submission_{args.model_name}_hidden_{args.hidden}_head_{args.num_heads}_f1_{round(avg_f1_score, 4)}).csv')

if __name__ == "__main__":
    main()
