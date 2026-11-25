import glob
import json
import os
import traceback

import numpy as np
import pandas as pd

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    base = None
    for cand in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        if "playground-series-s6e11" in cand:
            base = os.path.dirname(cand)
            break
    if base is None:
        raise FileNotFoundError("train.csv for playground-series-s6e11 not found")

    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "columns": list(train.columns),
        "target_rate": float(train["is_fraud"].mean()),
    }

    drop = {"id", "is_fraud"}
    feats = [c for c in train.columns if c not in drop]
    for df in (train, test):
        for c in feats:
            if df[c].dtype == object:
                df[c] = df[c].astype("category").cat.codes

    X = train[feats]
    y = train["is_fraud"].astype(int)
    X_test = test[feats]

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 40,
        "verbose": -1,
        "num_threads": 4,
        "seed": 42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dva = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx], reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=3000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(300, verbose=False)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_aucs.append(float(fold_auc))
        preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
        print(f"fold {fold}: auc={fold_auc:.5f} best_iter={model.best_iteration}", flush=True)

    oof_auc = float(roc_auc_score(y, oof))
    summary["fold_aucs"] = fold_aucs
    summary["oof_auc"] = oof_auc
    print("OOF AUC:", oof_auc, flush=True)

    preds = np.clip(preds, 0.0, 1.0)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": preds})
    assert len(sub) == len(test) and sub["is_fraud"].notna().all()
    sub.to_csv(os.path.join(WORK, "submission.csv"), index=False)
    with open(os.path.join(WORK, "summary.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2))
    print("submission written:", sub.shape, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(os.path.join(WORK, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
