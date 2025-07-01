import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='auc')
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    return model, score