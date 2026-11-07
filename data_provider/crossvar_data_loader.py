import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data_provider.data_processing import (
    align_two_site_series, append_encoder_proto_static, build_maml_static_feature,
    build_same_time_windows, clean_dataframe_columns, clean_site_name,
    compute_detrended_spearman, fit_crossvar_scalers_from_support,
    fit_static_scaler, get_common_sites, get_static_scaled, load_cluster_map,
    load_encoder_proto_static, load_site_meta, make_year_stamp,
    pack_maml_return, reconstruction_bounds, split_sites_fivefold,
)
from data_provider.relation_pretrain_loader import Dataset_RelationPretrain

warnings.filterwarnings('ignore')

class Dataset_CrossVar(Dataset):
    """Site-level reconstruction tasks with paired support/query windows: x[t:t+L] -> y[t:t+L]."""

    def __init__(self, root_path, flag='train', seq_len=64,
                 x_data_path=None,
                 y_data_path=None,
                 site_meta_path=None,
                 cluster_result_path=None,
                 support_ratio=None,
                 encoder_proto_path=None,
                 fold_id=None,
                 query_ratio=0.34,
                 verbose=True, adaptation_validation=False, window_stride=1, random_site_split=False):

        self.seq_len = seq_len
        if not 1 <= window_stride <= seq_len:
            raise ValueError('window_stride must be between 1 and seq_len')
        self.window_stride = window_stride
        self.random_site_split = random_site_split

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.root_path = root_path
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.site_meta_path = site_meta_path
        self.cluster_result_path = cluster_result_path


        self.support_ratio = support_ratio
        self.adaptation_validation = adaptation_validation
        self.query_ratio = query_ratio
        self.verbose = verbose
        self.encoder_proto_path = encoder_proto_path
        self.fold_id = fold_id
        self.encoder_proto_sim_map, self.encoder_proto_label_map, self.encoder_proto_dim = load_encoder_proto_static(
            self.encoder_proto_path
        )

        self.__read_data__()

    def __read_data__(self):
        # Load paired records.
        x_file = os.path.join(self.root_path, self.x_data_path)
        y_file = os.path.join(self.root_path, self.y_data_path)

        df_x_raw = pd.read_csv(x_file)
        df_y_raw = pd.read_csv(y_file)

        df_x_raw = clean_dataframe_columns(df_x_raw)
        df_y_raw = clean_dataframe_columns(df_y_raw)

        df_x_raw = df_x_raw.sort_values(by='year', ascending=True).reset_index(drop=True)
        df_y_raw = df_y_raw.sort_values(by='year', ascending=True).reset_index(drop=True)

        self.cluster_map = {} if self.random_site_split else load_cluster_map(self.cluster_result_path)
        self.site_meta_dict = load_site_meta(self.site_meta_path)

        common_sites = get_common_sites(df_x_raw, df_y_raw, year_col='year')

        # Split sites.
        if self.random_site_split:
            sites = sorted(common_sites)
            order = np.random.default_rng(2030).permutation(len(sites))
            sites = [sites[i] for i in order]
            n_train, n_val = int(len(sites) * 0.6), int(len(sites) * 0.2)
            train_sites, val_sites, test_sites = (
                sites[:n_train], sites[n_train:n_train + n_val], sites[n_train + n_val:])
        else:
            train_sites, val_sites, test_sites = split_sites_fivefold(
                common_sites, self.cluster_map, self.fold_id)

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites

        if self.verbose and self.set_type == 0:
            print(f'Sites: train={len(train_sites)}, val={len(val_sites)}, test={len(test_sites)}')

        self.current_sites = (train_sites, val_sites, test_sites)[self.set_type]

        # Fit scalers on training-site support histories.
        self.x_scaler = None
        self.y_scaler = None

        self.x_scaler, self.y_scaler = fit_crossvar_scalers_from_support(
            df_x_raw, df_y_raw, train_sites, self.seq_len,
            query_ratio=self.query_ratio)

        self.static_scaler = fit_static_scaler(
            train_sites=train_sites,
            site_meta_dict=self.site_meta_dict
        )

        # Apply global scaling.
        df_x_scaled = df_x_raw.copy()
        df_y_scaled = df_y_raw.copy()

        for site in common_sites:
            for frame, scaler in ((df_x_scaled, self.x_scaler),
                                  (df_y_scaled, self.y_scaler)):
                values = frame[[site]].values.astype(np.float32)
                frame.loc[:, site] = scaler.transform(values).flatten()

        self.static_feat_list = []
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.year_list = []

        self.site_cluster_map = {}
        self.support_len_list = []

        for site_idx, site in enumerate(self.current_sites):
            clean_s = clean_site_name(site)
            cluster_id = self.cluster_map.get(clean_s, -1)
            if not self.random_site_split:
                self.site_cluster_map[site_idx] = cluster_id

            years, x_vals, y_vals = align_two_site_series(
                df_x=df_x_scaled,
                df_y=df_y_scaled,
                site=site,
                year_col='year'
            )

            # Raw paired records for support correlation.
            _, x_raw, y_raw = align_two_site_series(
                df_x=df_x_raw,
                df_y=df_y_raw,
                site=site,
                year_col='year'
            )

            support_start, query_start = reconstruction_bounds(
                len(years), self.seq_len, self.query_ratio,
                self.support_ratio if self.flag == 'test' or self.adaptation_validation else None)
            self.support_len_list.append(query_start - support_start)
            years, x_vals, y_vals = (v[support_start:] for v in (years, x_vals, y_vals))

            # Compute correlation on support only.
            hist_corr = compute_detrended_spearman(
                x=x_raw[support_start:query_start],
                y=y_raw[support_start:query_start],
                min_len=30
            )

            static_scaled = get_static_scaled(
                site,
                self.site_meta_dict,
                self.static_scaler
            )

            # [lat, sin_lon, cos_lon, elevation, hist_corr]
            static_feat = np.concatenate(
                [static_scaled, np.array([hist_corr], dtype=np.float32)],
                axis=0
            )

            data_stamp = make_year_stamp(years)

            self.static_feat_list.append(static_feat.astype(np.float32))
            self.data_x_list.append(x_vals.astype(np.float32))
            self.data_y_list.append(y_vals.astype(np.float32))
            self.data_stamp_list.append(data_stamp.astype(np.float32))
            self.year_list.append(years.astype(np.float32))

    def __getitem__(self, index):
        # One item is one site/task.
        site_idx = index

        site_name = self.current_sites[site_idx]

        data_x = self.data_x_list[site_idx]
        data_y = self.data_y_list[site_idx]
        data_stamp = self.data_stamp_list[site_idx]
        years = self.year_list[site_idx]

        site_support_len = self.support_len_list[site_idx]

        # Build contemporaneous support/query windows.
        data_x_model, data_y_model = data_x, data_y
        support_x = data_x_model[:site_support_len]
        mu = support_x.mean(axis=0, keepdims=True)
        std = np.sqrt(support_x.var(axis=0, keepdims=True) + 1e-5)
        data_x_model = ((data_x_model - mu) / std).astype(np.float32)

        windows = build_same_time_windows(
            data_x=data_x_model,
            data_y=data_y_model,
            data_stamp=data_stamp,
            years=years,
            seq_len=self.seq_len,
            support_len=site_support_len,
            stride=self.window_stride
        )

        static_feat = self.static_feat_list[site_idx]

        # Local statistics use only the selected support.
        local_series = data_x[:site_support_len].flatten()

        maml_static_feat = build_maml_static_feature(
            static_feat=static_feat,
            local_series=local_series,
        )

        maml_static_feat = append_encoder_proto_static(
            maml_static_feat=maml_static_feat,
            site_name=site_name,
            proto_sim_map=self.encoder_proto_sim_map,
            proto_dim=self.encoder_proto_dim,
            proto_label_map=self.encoder_proto_label_map,
        )

        return pack_maml_return(
            windows=windows,
            maml_static_feat=maml_static_feat
        )

    def __len__(self):
        return len(self.current_sites)

    def inverse_transform(self, data, var='y'):
        """Restore original units with the scaler for x or y (default)."""

        if var == 'x':
            if self.x_scaler is not None:
                return self.x_scaler.inverse_transform(data)
            return data

        else:
            if self.y_scaler is not None:
                return self.y_scaler.inverse_transform(data)
            return data

class Dataset_FCR_CrossVar(Dataset_CrossVar):
    """One simulation per task; day indices use the existing year field."""

    def __init__(self, root_path, x_data_path, y_data_path, flag='train',
                 seq_len=32, support_ratio=None, encoder_proto_path=None,
                 query_ratio=0.34, adaptation_validation=False, window_stride=1):
        self.seq_len = seq_len
        if not 1 <= window_stride <= seq_len:
            raise ValueError('window_stride must be between 1 and seq_len')
        self.window_stride = window_stride
        x = pd.read_csv(os.path.join(root_path, x_data_path)).set_index('year').sort_index()
        y = pd.read_csv(os.path.join(root_path, y_data_path)).set_index('year').sort_index()
        sites = [f'FCR_{i:04d}' for i in range(1, 51)]
        order = np.random.default_rng(2030).permutation(len(sites))
        shuffled = [sites[i] for i in order]
        n_train, n_val = int(len(sites) * 0.6), int(len(sites) * 0.2)
        self.train_sites = shuffled[:n_train]
        self.val_sites = shuffled[n_train:n_train + n_val]
        self.test_sites = shuffled[n_train + n_val:]
        self.current_sites = {'train': self.train_sites, 'val': self.val_sites,
                              'test': self.test_sites,
                              'pretrain': sites,
                              'descriptor': sites}[flag]
        support_start, query_start = reconstruction_bounds(
            len(x), self.seq_len, query_ratio,
            support_ratio if flag in ('test', 'descriptor') or adaptation_validation else None)
        self.support_len = query_start - support_start
        fit_sites = (sites if flag in ('pretrain', 'descriptor')
                     else self.train_sites)
        self.x_scaler = StandardScaler().fit(x[fit_sites].iloc[:query_start].to_numpy().reshape(-1, 1))
        self.y_scaler = StandardScaler().fit(y[fit_sites].iloc[:query_start].to_numpy().reshape(-1, 1))
        self.data_x_list = [self.x_scaler.transform(x[[site]].iloc[support_start:]).astype(np.float32)
                            for site in self.current_sites]
        self.data_y_list = [self.y_scaler.transform(y[[site]].iloc[support_start:]).astype(np.float32)
                            for site in self.current_sites]
        self.year_list = [x.index[support_start:].to_numpy(dtype=np.float32)] * len(self.current_sites)
        self.data_stamp_list = [make_year_stamp(x.index[support_start:])] * len(self.current_sites)
        self.support_len_list = [self.support_len] * len(self.current_sites)
        self.site_cluster_map = {}  # Random scenario split; no cluster-balanced sampling.
        if flag in ('pretrain', 'descriptor'):
            self.static_feat_list = [np.empty(0, dtype=np.float32)] * len(self.current_sites)
        else:
            self.static_feat_list = []
            sim_map, label_map, proto_dim = load_encoder_proto_static(encoder_proto_path)
            for i, site in enumerate(self.current_sites):
                # Only support history contributes observed x/y features.
                hist_corr = compute_detrended_spearman(
                    x[site].iloc[support_start:query_start].to_numpy(),
                    y[site].iloc[support_start:query_start].to_numpy(), min_len=30,
                )
                static = build_maml_static_feature(
                    static_feat=np.array([hist_corr], dtype=np.float32),
                    local_series=self.data_x_list[i][:self.support_len].flatten(),
                        )
                static = append_encoder_proto_static(
                    static, site, sim_map, proto_dim,
                    proto_label_map=label_map,
                )
                self.static_feat_list.append(static)

    def __getitem__(self, index):
        x, y = self.data_x_list[index], self.data_y_list[index]
        support = x[:self.support_len]
        x = ((x - support.mean(0)) / np.sqrt(support.var(0) + 1e-5)).astype(np.float32)
        windows = build_same_time_windows(
            x, y, self.data_stamp_list[index], self.year_list[index],
            self.seq_len, self.support_len, stride=self.window_stride)
        return pack_maml_return(windows, self.static_feat_list[index])

class Dataset_FCR_RelationPretrain(Dataset_RelationPretrain):
    """Reuse relation window sampling on every scenario's support; no static prefix."""

    def __read_data__(self):
        data = Dataset_FCR_CrossVar(
            self.root_path, self.x_data_path, self.y_data_path, flag='pretrain',
            seq_len=self.seq_len, query_ratio=self.query_ratio)
        self.site_names = data.current_sites
        self.num_sites = len(self.site_names)
        self.data_x_list, self.data_y_list = [], []
        for values, target in ((data.data_x_list, self.data_x_list),
                               (data.data_y_list, self.data_y_list)):
            for series in values:
                history = series[:data.support_len]
                target.append(((history - history.mean(0)) /
                               np.sqrt(history.var(0) + 1e-5)).astype(np.float32))
        self.data_stamp_list = [make_year_stamp(years[:data.support_len])
                                for years in data.year_list]
        self.static_feat_list = data.static_feat_list
