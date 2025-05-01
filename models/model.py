# model.py
from lightgbm import LGBMClassifier

# 모델 정의 함수 예시
def get_model(params=None):
    if params is None:
        params = {
            'objective': 'multiclass',
            'num_class': 3,  # 예시 클래스 수
            'random_state': 42,
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'max_depth': -1,
            'n_jobs': -1,
            'verbosity': -1
        }
    model = LGBMClassifier(**params)
    return model
