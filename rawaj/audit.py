"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations


import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.metrics import roc_auc_score


def _is_missing(value):
    """Treat NaN, None, and blank strings as missing."""
    if pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before training a model."""

    # 1. Check that the required columns exist.
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)

    # 2. Identify missing values, including blank strings.
    missing_masks = {
        col: df[col].map(_is_missing)
        for col in df.columns
    }

    missing = {
        col: float(mask.mean()) if len(df) else 0.0
        for col, mask in missing_masks.items()
    }

    # 3. Count duplicate IDs, ignoring missing IDs.
    valid_ids = df.loc[~missing_masks[id_col], id_col]
    duplicate_ids = sorted(
        valid_ids[valid_ids.duplicated(keep=False)].unique().tolist()
    )

    # 4. Count rows that repeat an earlier row exactly.
    duplicate_rows = int(df.duplicated().sum())

    # 5. Find constant columns, excluding the target.
    constant_columns = []

    for col in df.columns:
        if col == target:
            continue

        non_missing = df.loc[~missing_masks[col], col]

        if non_missing.nunique() <= 1:
            constant_columns.append(col)

    constant_columns.sort()

    # 6. Count rows without a target label.
    labelled_mask = ~missing_masks[target]
    labelled = df.loc[labelled_mask, target]

    unlabelled_rows = int(missing_masks[target].sum())

    # 7. Calculate the positive rate among labelled rows.
    if len(labelled):
        positive_rate = float((labelled == 1).mean())
    else:
        positive_rate = 0.0

    # 8. Parse timestamps and find invalid non-missing values.
    parsed_times = pd.to_datetime(
        df[time_col].where(~missing_masks[time_col]),
        errors="coerce",
    )

    bad_timestamps = int(
        (parsed_times.isna() & ~missing_masks[time_col]).sum()
    )

    valid_times = parsed_times.dropna()

    if len(valid_times):
        time_range = [
            valid_times.min().isoformat(),
            valid_times.max().isoformat(),
        ]
    else:
        time_range = [None, None]

    # 9. Look for possible data leakage.
    leakage_suspects = []
    labelled_df = df.loc[labelled_mask]
    y = (labelled_df[target] == 1).astype(int)

    if len(labelled_df) >= 20 and y.nunique() == 2:
        for col in df.columns:
            if col in (target, id_col, time_col):
                continue

            values = labelled_df[col]
            is_missing_col = missing_masks[col].loc[labelled_mask]

            # Count missingness as a distinct value for this rule.
            distinct_count = values.loc[~is_missing_col].nunique()
            if is_missing_col.any():
                distinct_count += 1

            # A column with only one distinct value is never a suspect.
            if distinct_count <= 1:
                continue

            suspect = False

            # Check (a): numeric or boolean values that predict the label.
            if is_numeric_dtype(values) or is_bool_dtype(values):
                non_missing_mask = ~is_missing_col
                values_present = values.loc[non_missing_mask]
                y_present = y.loc[non_missing_mask]

                if len(values_present) >= 20 and y_present.nunique() == 2:
                    auc = roc_auc_score(y_present, values_present)

                    if auc >= 0.98 or auc <= 0.02:
                        suspect = True

            # Check (b): whether missingness itself predicts the label.
            missing_indicator = is_missing_col.astype(int)

            if missing_indicator.nunique() > 1:
                missing_auc = roc_auc_score(y, missing_indicator)

                if missing_auc >= 0.98 or missing_auc <= 0.02:
                    suspect = True

            if suspect:
                leakage_suspects.append(col)

    leakage_suspects.sort()

    # 10. Return exactly the keys required by the task.
    return {
        "n_rows": int(len(df)),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
