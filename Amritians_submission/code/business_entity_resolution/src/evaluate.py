import numpy as np

def compute_s1_f05(pred_ids: set, true_ids: set) -> float:
    """
    Compute F_0.5 score for a single Source 1 entity.
    """
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    
    if len(pred_ids) == 0:
        return 0.0
    
    tp = len(pred_ids.intersection(true_ids))
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    
    num = 1.25 * precision * recall
    denom = 0.25 * precision + recall
    return num / denom if denom > 0 else 0.0

def evaluate_predictions(pred_dict: dict, gt_dict: dict) -> dict:
    """
    Macro-averaged F_0.5 score across all S1 entities in gt_dict.
    
    pred_dict: {s1_id: set/list of predicted S2/S3 entity_ids}
    gt_dict:   {s1_id: set/list of true matching S2/S3 entity_ids}
    """
    scores = []
    precisions = []
    recalls = []
    
    for s1_id, true_list in gt_dict.items():
        true_set = set(true_list) if not isinstance(true_list, set) else true_list
        pred_list = pred_dict.get(s1_id, set())
        pred_set = set(pred_list) if not isinstance(pred_list, set) else pred_list
        
        score = compute_s1_f05(pred_set, true_set)
        scores.append(score)
        
        # Calculate P and R for reporting
        if len(true_set) == 0:
            p = 1.0 if len(pred_set) == 0 else 0.0
            r = 1.0 if len(pred_set) == 0 else 0.0
        elif len(pred_set) == 0:
            p = 0.0
            r = 0.0
        else:
            tp = len(pred_set.intersection(true_set))
            p = tp / len(pred_set)
            r = tp / len(true_set)
        precisions.append(p)
        recalls.append(r)
        
    macro_f05 = float(np.mean(scores))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    
    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "num_entities": len(gt_dict)
    }

if __name__ == "__main__":
    # Test example from prompt:
    # Pred: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    # Ground Truth: S1-00001 -> [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.71428...
    sample_gt = {"S1-00001": ["S2-00047", "S3-00812"]}
    sample_pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    res = evaluate_predictions(sample_pred, sample_gt)
    print("Test Sample Evaluation Result:", res)
    assert abs(res["macro_f05"] - 0.7142857) < 1e-4, "Evaluation formula test failed!"
    print("evaluate.py PASSED validation!")
