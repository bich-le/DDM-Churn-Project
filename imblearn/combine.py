from __future__ import annotations

import numpy as np


class SMOTETomek:
    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        try:
            import pandas as pd
        except Exception:  # pragma: no cover
            pd = None

        if pd is not None and isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = None
            X_arr = np.asarray(X)

        if pd is not None and isinstance(y, (pd.Series, pd.DataFrame)):
            y_series = y.squeeze().copy()
        else:
            y_series = np.asarray(y)

        y_values = np.asarray(y_series)
        classes, counts = np.unique(y_values, return_counts=True)
        if len(classes) != 2:
            return X, y

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        majority_count = counts.max()
        minority_indices = np.flatnonzero(y_values == minority_class)

        if minority_indices.size == 0:
            return X, y

        rng = np.random.default_rng(self.random_state)
        sampled_indices = rng.choice(
            minority_indices, size=majority_count, replace=True
        )

        if X_df is not None:
            minority_frame = X_df.iloc[sampled_indices].reset_index(drop=True)
            if pd is not None and isinstance(y_series, pd.Series):
                minority_target = y_series.iloc[sampled_indices].reset_index(drop=True)
                majority_frame = X_df.iloc[
                    np.flatnonzero(y_values == majority_class)
                ].reset_index(drop=True)
                majority_target = y_series.iloc[
                    np.flatnonzero(y_values == majority_class)
                ].reset_index(drop=True)
                X_resampled = pd.concat(
                    [majority_frame, minority_frame], ignore_index=True
                )
                y_resampled = pd.concat(
                    [majority_target, minority_target], ignore_index=True
                )
            else:
                majority_frame = X_df.iloc[
                    np.flatnonzero(y_values == majority_class)
                ].reset_index(drop=True)
                X_resampled = pd.concat(
                    [majority_frame, minority_frame], ignore_index=True
                )
                y_resampled = np.concatenate(
                    [y_values[y_values == majority_class], y_values[sampled_indices]]
                )
            return X_resampled, y_resampled

        majority_array = X_arr[y_values == majority_class]
        minority_array = X_arr[sampled_indices]
        X_resampled = np.concatenate([majority_array, minority_array], axis=0)
        y_resampled = np.concatenate(
            [y_values[y_values == majority_class], y_values[sampled_indices]]
        )
        return X_resampled, y_resampled
