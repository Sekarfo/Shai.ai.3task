#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

# Фиксируем фичи, доступные на инференсе
DESIRED_NUM_COLS = ["sla_days", "storm_flag", "fx_volatility_7d", "dow", "month"]
TARGET_COL = "delay_flag"

def iso_dow(series_date: pd.Series) -> pd.Series:

    return series_date.dt.weekday.add(1)

def prepare_df(df: pd.DataFrame):
    df = df.copy()

    # 1) Дата и производные признаки
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "eta" in c.lower():
            try:
                dt = pd.to_datetime(df[c], errors="coerce", utc=True)
                if dt.notna().any():
                    date_col = c
                    df[c] = dt
                
                    df["dow"] = iso_dow(dt)
                    df["month"] = dt.dt.month
                    break
            except Exception:
                pass

    # 2) Таргет
    target_col = None
    if TARGET_COL in df.columns:
        target_col = TARGET_COL
    elif {"actual_days","sla_days"}.issubset(df.columns):
        df[TARGET_COL] = (pd.to_numeric(df["actual_days"], errors="coerce") >
                          pd.to_numeric(df["sla_days"], errors="coerce")).astype("Int64").astype(int)
        target_col = TARGET_COL
    else:
        raise SystemExit("Нужен таргет: 'delay_flag' или пара 'actual_days' + 'sla_days'.")


    if "storm_flag" in df.columns:
        df["storm_flag"] = pd.to_numeric(df["storm_flag"], errors="coerce").fillna(0).clip(0,1).astype(int)
    else:
        df["storm_flag"] = 0

    
    for c in ["sla_days", "fx_volatility_7d", "dow", "month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    
    num_cols = [c for c in DESIRED_NUM_COLS if c in df.columns]

    # Проверки
    missing = [c for c in DESIRED_NUM_COLS if c not in num_cols]
    if missing:
        print(f"[WARN] В датасете нет фич: {missing}. Они будут иммутацией NaN → median.")

    # Дропаут по таргету и критичным фичам
    keep_cols = list(set(num_cols + [target_col] + ( [date_col] if date_col else [] )))
    df = df[keep_cols].copy()
    df = df.dropna(subset=[target_col])  # таргет обязателен

    return df, num_cols, target_col, date_col

def time_aware_split(df: pd.DataFrame, target_col: str, date_col: str | None):
    if date_col:
        df = df.sort_values(by=date_col)
        split = int(0.8 * len(df))
        train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
        # если тест пуст — fallback
        if len(test_df) == 0 or len(train_df) == 0:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return train_df, test_df

def fit_and_eval(train_df, test_df, num_cols, target_col, outdir: Path):
    X_train, y_train = train_df[num_cols], train_df[target_col].astype(int)
    X_test,  y_test  = test_df[num_cols],  test_df[target_col].astype(int)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    # Подбор порога по F1
    p, r, thr = precision_recall_curve(y_test, proba)
    f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(thr[best_idx-1]) if best_idx > 0 and best_idx-1 < len(thr) else 0.5
    # Жёсткая шкала под бизнес-логику (можно править)
    thresholds = {
        "yellow": float(max(0.40, min(0.60, best_thr))),
        "red": 0.60
    }

    # Сохраняем артефакты
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "model.pkl")
    (outdir / "num_cols.json").write_text(json.dumps(num_cols, ensure_ascii=False, indent=2))
    (outdir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2))

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "num_cols": num_cols,
        "metrics": {"roc_auc": float(roc), "pr_auc": float(pr), "brier": float(brier)},
        "best_threshold_f1": best_thr,
        "thresholds_used": thresholds
    }
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main():
    # Указываем пути к файлам и папкам вручную
    csv_path = r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\data\shipping_dataset_expanded.csv"
    outdir = Path(r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\artifacts")
    df = pd.read_csv(csv_path)

    df, num_cols, target_col, date_col = prepare_df(df)
    train_df, test_df = time_aware_split(df, target_col, date_col)
    fit_and_eval(train_df, test_df, num_cols, target_col, outdir)

if __name__ == "__main__":
    main()
