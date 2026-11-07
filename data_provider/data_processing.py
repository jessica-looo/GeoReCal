import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.signal import detrend

warnings.filterwarnings('ignore')

def clean_site_name(name):
    """Normalize site names across CSV headers, cluster labels, and metadata."""
    return (
        str(name)
        .strip()
        .strip('"')
        .strip("'")
        .replace('  ', ' ')
    )

def clean_dataframe_columns(df):
    """Normalize dataframe column names."""
    return df.rename(columns={c: clean_site_name(c) for c in df.columns})

def load_cluster_map(cluster_result_path, cluster_col='Level_0'):
    """Return {site_name: cluster_id} from the cluster CSV."""
    cluster_df = pd.read_csv(cluster_result_path)

    site_col = cluster_df.columns[0]
    cluster_df[site_col] = cluster_df[site_col].apply(clean_site_name)

    if cluster_col not in cluster_df.columns:
        raise ValueError(
            f"cluster_col={cluster_col} not found in {cluster_result_path}. "
            f"Available columns: {cluster_df.columns.tolist()}"
        )

    return cluster_df.set_index(site_col)[cluster_col].to_dict()

def load_site_meta(site_meta_path):
    """Return per-site Latitude, Longitude, and Elevation metadata."""
    df_meta = pd.read_csv(site_meta_path)
    df_meta['site_name_clean'] = df_meta['site'].apply(clean_site_name)

    required_cols = ['Latitude', 'Longitude', 'Elevation']
    for col in required_cols:
        if col not in df_meta.columns:
            raise ValueError(f"{col} not found in site metadata: {site_meta_path}")

    return df_meta.set_index('site_name_clean')[required_cols].to_dict('index')

def get_static_raw(site, site_meta_dict):
    """Return [latitude, sin(longitude), cos(longitude), elevation].
    Missing metadata defaults to [0, 0, 1, 0].
    """
    clean_s = clean_site_name(site)
    meta = site_meta_dict.get(
        clean_s,
        {'Latitude': 0.0, 'Longitude': 0.0, 'Elevation': 0.0}
    )

    lat = float(meta.get('Latitude', 0.0))
    lon = float(meta.get('Longitude', 0.0))
    elev = float(meta.get('Elevation', 0.0))

    lon_rad = np.radians(lon)
    return np.array([lat, np.sin(lon_rad), np.cos(lon_rad), elev], dtype=np.float32)

def fit_static_scaler(train_sites, site_meta_dict):
    """
    Fit static scaler only on train_sites.
    """
    values = np.asarray([get_static_raw(s, site_meta_dict) for s in train_sites],
                        dtype=np.float32)
    return StandardScaler().fit(values)

def get_static_scaled(site, site_meta_dict, static_scaler):
    """
    Return scaled static feature with shape [4].
    """
    static_raw = get_static_raw(site, site_meta_dict).reshape(1, -1)
    return static_scaler.transform(static_raw)[0].astype(np.float32)

def get_common_sites(df_a, df_b=None, year_col='year'):
    """Return sorted site columns, intersected with df_b when provided."""
    sites_a = set(df_a.columns) - {year_col}

    if df_b is None:
        return sorted(sites_a)

    sites_b = set(df_b.columns) - {year_col}
    return sorted(sites_a & sites_b)

def split_sites_fivefold(valid_sites, cluster_map, fold_id, seed=2030):
    if fold_id not in range(5):
        raise ValueError('fold_id must be 0 through 4')
    sites = sorted(valid_sites)
    if len(sites) < 5 or len(sites) != len(set(sites)):
        raise ValueError('Five-fold splitting requires at least five distinct sites')
    if any(cluster_map.get(clean_site_name(s)) not in range(3) for s in sites):
        raise ValueError('Every valid site requires a cluster label in 0, 1, 2')
    groups = [[] for _ in range(5)]
    rng = np.random.RandomState(seed)
    offset = 0
    for cid in range(3):
        pool = [s for s in sites if cluster_map[clean_site_name(s)] == cid]
        rng.shuffle(pool)
        for i, site in enumerate(pool):
            groups[(offset + i) % 5].append(site)
        offset = (offset + len(pool)) % 5
    val_id = (fold_id + 1) % 5
    train = [s for i, group in enumerate(groups)
             if i not in (fold_id, val_id) for s in group]
    return train, groups[val_id], groups[fold_id]


def reconstruction_bounds(total_len, seq_len, query_ratio=0.34, support_ratio=None):
    """Return support start and fixed query start; ratios refer to the full record."""
    if not 0 < query_ratio < 1:
        raise ValueError('query_ratio must be between 0 and 1')
    query_start = total_len - int(total_len * query_ratio)
    if support_ratio is not None and not 0 < support_ratio <= 1:
        raise ValueError('support_ratio must be between 0 and 1')
    support_len = query_start if support_ratio is None else int(total_len * support_ratio)
    if support_len > query_start or min(support_len, total_len - query_start) < seq_len:
        raise ValueError('Support/query must not overlap and must each contain a full window')
    return query_start - support_len, query_start

def fit_crossvar_scalers_from_support(df_x, df_y, sites, seq_len, query_ratio):
    """Fit global scalers on complete pre-query history, in site order."""
    support_len = reconstruction_bounds(len(df_x), seq_len, query_ratio)[1]
    values = [np.vstack([df[[site]].values[:support_len].astype(np.float32)
                         for site in sites]) for df in (df_x, df_y)]
    return tuple(StandardScaler().fit(v) for v in values)


def make_year_stamp(years):
    """Return standardized time coordinates [T, 1]."""
    years = np.asarray(years, dtype=np.float32)

    mean = years.mean()
    std = years.std()

    if std < 1e-6:
        std = 1.0

    year_norm = (years - mean) / std
    return year_norm.reshape(-1, 1).astype(np.float32)

def align_two_site_series(df_x, df_y, site, year_col='year'):
    """
    Extract a site from already aligned, complete x/y tables.

    Return:
        years, x_vals, y_vals
    where:
        years shape [T]
        x_vals shape [T, 1]
        y_vals shape [T, 1]
    """
    return (df_x[year_col].to_numpy(dtype=np.float32),
            df_x[[site]].to_numpy(dtype=np.float32),
            df_y[[site]].to_numpy(dtype=np.float32))


def compute_detrended_spearman(x, y, min_len=30):
    """
    Compute detrended Spearman correlation.
    Return 0.0 if data is too short or invalid.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = min(len(x), len(y))
    if n < min_len:
        return 0.0

    x = x[:n]
    y = y[:n]

    try:
        x_dt = detrend(x, type='linear')
        y_dt = detrend(y, type='linear')
        rho, _ = spearmanr(x_dt, y_dt)

        if np.isnan(rho):
            return 0.0

        return float(rho)

    except Exception:
        return 0.0

def compute_local_stats(series):
    """Return mean, standard deviation, and low-to-high frequency power ratio."""
    x = np.asarray(series).flatten().astype(np.float32)

    local_mean = float(np.mean(x))
    local_std = float(np.std(x) + 1e-6)

    centered = x - local_mean
    fft_power = np.abs(np.fft.rfft(centered)) ** 2

    if len(fft_power) <= 1:
        local_f0 = 0.0
    else:
        split = max(1, len(fft_power) // 5)
        low_power = fft_power[:split].sum()
        high_power = fft_power[split:].sum()
        local_f0 = float(low_power / (high_power + 1e-8))

    return local_mean, local_std, local_f0

def load_encoder_proto_static(encoder_proto_path):
    """Load relation-pretraining prototype similarities and cluster labels.
    Expected fields: site_names [N], site_sim [N, K], optional site_cluster_label [N].
    Return similarity and label maps keyed by site, plus prototype count K.
    """
    if encoder_proto_path is None:
        return {}, {}, 0

    if not os.path.exists(encoder_proto_path):
        raise FileNotFoundError(f"encoder_proto_path not found: {encoder_proto_path}")

    data = np.load(encoder_proto_path, allow_pickle=True)
    if 'site_names' not in data or 'site_sim' not in data:
        raise ValueError(
            f"{encoder_proto_path} must contain 'site_names' and 'site_sim'. "
            f"Available keys: {list(data.keys())}"
        )

    site_names = [clean_site_name(s) for s in data['site_names'].tolist()]
    site_sim = np.asarray(data['site_sim'], dtype=np.float32)

    if site_sim.ndim != 2:
        raise ValueError(f"site_sim must be 2D [N, K], got shape={site_sim.shape}")

    if len(site_names) != site_sim.shape[0]:
        raise ValueError(
            f"site_names length ({len(site_names)}) does not match site_sim rows ({site_sim.shape[0]})"
        )

    sim_map = {
        site: site_sim[i].astype(np.float32)
        for i, site in enumerate(site_names)
    }

    label_map = {}
    if 'site_cluster_label' in data:
        labels = np.asarray(data['site_cluster_label'], dtype=np.int64).reshape(-1)
        if len(labels) != len(site_names):
            raise ValueError(
                f"site_cluster_label length ({len(labels)}) does not match "
                f"site_names length ({len(site_names)})"
            )
        label_map = {
            site: int(labels[i])
            for i, site in enumerate(site_names)
        }

    return sim_map, label_map, int(site_sim.shape[1])

def append_encoder_proto_static(
    maml_static_feat,
    site_name,
    proto_sim_map,
    proto_dim,
    proto_label_map=None,
):
    """
    Append offline encoder prototype cosine similarity and cluster one-hot.
    Missing sites receive zeros so val/test can still run.
    """
    base = np.asarray(maml_static_feat, dtype=np.float32).flatten()

    if proto_dim <= 0:
        return base

    clean_s = clean_site_name(site_name)
    sim = proto_sim_map.get(clean_s)
    if sim is None:
        sim = np.zeros(proto_dim, dtype=np.float32)
    else:
        sim = np.asarray(sim, dtype=np.float32).flatten()
        if sim.shape[0] != proto_dim:
            fixed = np.zeros(proto_dim, dtype=np.float32)
            take = min(proto_dim, sim.shape[0])
            fixed[:take] = sim[:take]
            sim = fixed

    extra = [base, sim]

    onehot = np.zeros(proto_dim, dtype=np.float32)
    if proto_label_map is None:
        label = None
    else:
        label = proto_label_map.get(clean_s)
    if label is not None and 0 <= int(label) < proto_dim:
        onehot[int(label)] = 1.0
    extra.append(onehot)

    return np.concatenate(extra, axis=0).astype(np.float32)

def build_same_time_windows(
    data_x,
    data_y,
    data_stamp,
    years,
    seq_len,
    support_len,
    stride=1,
):
    """Build paired x/y windows of length seq_len with the given stride.
    Query windows start at support_len; support and query do not overlap.
    """
    total_len = len(data_x)
    window_size = seq_len

    support_x, support_y = [], []
    support_x_mark, support_y_mark = [], []

    for i in range(0, support_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        support_x.append(data_x[s_begin:s_end])
        support_y.append(data_y[s_begin:s_end])
        support_x_mark.append(data_stamp[s_begin:s_end])
        support_y_mark.append(data_stamp[s_begin:s_end])

    query_x, query_y = [], []
    query_x_mark, query_y_mark = [], []
    query_years = []

    q_start = support_len

    for i in range(q_start, total_len - window_size + 1, stride):
        s_begin = i
        s_end = s_begin + seq_len

        query_x.append(data_x[s_begin:s_end])
        query_y.append(data_y[s_begin:s_end])
        query_x_mark.append(data_stamp[s_begin:s_end])
        query_y_mark.append(data_stamp[s_begin:s_end])

        query_years.append(years[s_begin])

    return {
        'support_x': support_x,
        'support_y': support_y,
        'support_x_mark': support_x_mark,
        'support_y_mark': support_y_mark,
        'query_x': query_x,
        'query_y': query_y,
        'query_x_mark': query_x_mark,
        'query_y_mark': query_y_mark,
        'query_years': query_years,
    }

def build_maml_static_feature(static_feat, local_series):
    local_mean, local_std, local_f0 = compute_local_stats(local_series)
    return np.concatenate((np.asarray(static_feat).flatten(),
                           [local_mean, local_std, local_f0])).astype(np.float32)

def to_float_tensor(x):
    """
    Convert list/array to torch.float32 tensor.
    """
    return torch.tensor(np.asarray(x), dtype=torch.float32)

def pack_maml_return(windows, maml_static_feat):
    """Support x/y/marks, query x/y/marks, static context and query years."""
    return tuple(to_float_tensor(windows[key]) for key in (
        'support_x', 'support_y', 'support_x_mark', 'support_y_mark',
        'query_x', 'query_y', 'query_x_mark', 'query_y_mark',
    )) + (to_float_tensor(maml_static_feat), to_float_tensor(windows['query_years']))
