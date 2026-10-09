import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from evaluate import evaluate_predictions, compute_s1_f05

def train_lightgbm_matcher(X_train: pd.DataFrame, y_train: np.ndarray,
                           X_val: pd.DataFrame, y_val: np.ndarray,
                           val_pairs_df: pd.DataFrame, val_gt_map: dict):
    """
    Train LightGBM binary classifier, perform probability threshold search,
    and return trained model + optimal threshold + validation evaluation metrics.
    """
    print(f"Training LightGBM matcher on {len(X_train)} candidate pairs...", flush=True)
    print(f"Train positive ratio: {np.mean(y_train):.4f}", flush=True)
    print(f"Val positive ratio: {np.mean(y_val):.4f}", flush=True)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'is_unbalance': True,
        'random_state': config.SEED,
        'n_jobs': config.N_JOBS,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=50)]
    )

    # Feature Importance
    feature_imp = pd.Series(model.feature_importance(importance_type='gain'), index=X_train.columns).sort_values(ascending=False)
    print("\n--- TOP 10 FEATURE IMPORTANCES (GAIN) ---")
    print(feature_imp.head(10))

    # Predict probabilities on Validation Candidate Pairs
    print("\n--- OPTIMIZING PROBABILITY THRESHOLD FOR MACRO F_0.5 ---")
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    val_df = val_pairs_df.copy()
    val_df['prob'] = val_probs

    best_thresh = 0.5
    best_f05 = -1.0
    best_eval = {}

    threshold_grid = np.linspace(0.10, 0.90, 81)
    
    # Pre-group validation predictions by S1 ID
    grouped_val = val_df.groupby('source1_entity_id')

    for thresh in threshold_grid:
        pred_map = {}
        for s1_id, group in grouped_val:
            passed = group[group['prob'] >= thresh]
            if not passed.empty:
                pred_map[s1_id] = passed['candidate_entity_id'].tolist()
        
        eval_res = evaluate_predictions(pred_map, val_gt_map)
        f05 = eval_res['macro_f05']
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_eval = eval_res

    print(f"Optimal Probability Threshold: {best_thresh:.4f}")
    print(f"Validation Macro F_0.5 Score: {best_f05:.4f}")
    print(f"Validation Precision: {best_eval['macro_precision']:.4f}")
    print(f"Validation Recall: {best_eval['macro_recall']:.4f}")

    return model, best_thresh, best_eval, feature_imp

if __name__ == "__main__":
    print("train_matcher.py loaded successfully.")
