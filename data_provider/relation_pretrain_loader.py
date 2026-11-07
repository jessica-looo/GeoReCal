import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.data_processing import (
    align_two_site_series,
    clean_dataframe_columns,
    clean_site_name,
    fit_crossvar_scalers_from_support,
    fit_static_scaler,
    get_common_sites,
    get_static_scaled,
    load_site_meta,
    make_year_stamp,
    reconstruction_bounds,
)

class Dataset_RelationPretrain(Dataset):
    """Sample two paired history windows per site with geographic context."""

    def __init__(
        self,
        root_path,
        x_data_path,
        y_data_path,
        seq_len,
        samples_per_site=8,
        site_meta_path=None,
        verbose=True,
        query_ratio=0.34,
    ):
        self.root_path = root_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.seq_len = int(seq_len)
        self.samples_per_site = int(samples_per_site)
        self.query_ratio = query_ratio
        self.site_meta_path = site_meta_path
        self.verbose = verbose

        self.__read_data__()

    def __read_data__(self):
        x_file = os.path.join(self.root_path, self.x_data_path)
        y_file = os.path.join(self.root_path, self.y_data_path)

        df_x_raw = clean_dataframe_columns(pd.read_csv(x_file))
        df_y_raw = clean_dataframe_columns(pd.read_csv(y_file))
        df_x_raw = df_x_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        df_y_raw = df_y_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        common_sites = get_common_sites(df_x_raw, df_y_raw, year_col='year')

        self.site_meta_dict = load_site_meta(self.site_meta_path)
        self.x_scaler, self.y_scaler = fit_crossvar_scalers_from_support(
            df_x_raw, df_y_raw, common_sites, self.seq_len, self.query_ratio)
        self.static_scaler = fit_static_scaler(common_sites, self.site_meta_dict)

        df_x_scaled = df_x_raw.copy()
        df_y_scaled = df_y_raw.copy()
        for site in common_sites:
            for frame, scaler in ((df_x_scaled, self.x_scaler),
                                  (df_y_scaled, self.y_scaler)):
                values = frame[[site]].values.astype(np.float32)
                frame.loc[:, site] = scaler.transform(values).flatten()

        self.site_names = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.static_feat_list = []

        history_len = reconstruction_bounds(len(df_x_raw), self.seq_len, self.query_ratio)[1]
        for site in common_sites:
            years, x_vals, y_vals = align_two_site_series(df_x_scaled, df_y_scaled, site, year_col='year')

            static_scaled = get_static_scaled(site, self.site_meta_dict, self.static_scaler)
            # Geographic features: scaled latitude and longitude encoding.
            static_feat = static_scaled[:3].astype(np.float32)

            self.site_names.append(clean_site_name(site))
            for values, target in (
                (x_vals, self.data_x_list),
                (y_vals, self.data_y_list),
            ):
                history = values[:history_len]
                mu = history.mean(axis=0, keepdims=True)
                std = np.sqrt(history.var(axis=0, keepdims=True) + 1e-5)
                target.append(((history - mu) / std).astype(np.float32))
            self.data_stamp_list.append(make_year_stamp(years[:history_len]).astype(np.float32))
            self.static_feat_list.append(static_feat)

        self.num_sites = len(self.site_names)
        if self.verbose:
            print(
                f"Relation pretrain sites: {self.num_sites}"
            )

    def _sample_window(self, site_id, rng):
        x = self.data_x_list[site_id]
        y = self.data_y_list[site_id]
        stamp = self.data_stamp_list[site_id]
        max_start = len(x) - self.seq_len
        start = int(rng.randint(0, max_start + 1)) if max_start > 0 else 0
        end = start + self.seq_len
        return x[start:end], y[start:end], stamp[start:end]

    def __getitem__(self, index):
        site_id = index % self.num_sites
        rng = np.random.RandomState((index + 1) * 1009 + np.random.randint(0, 1000000))

        x1, y1, m1 = self._sample_window(site_id, rng)
        x2, y2, m2 = self._sample_window(site_id, rng)

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(m1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(m2, dtype=torch.float32),
            torch.tensor(self.static_feat_list[site_id], dtype=torch.float32),
            torch.tensor(site_id, dtype=torch.long),
        )

    def __len__(self):
        return self.num_sites * max(1, self.samples_per_site)
