from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch
import collections

from data_provider.crossvar_data_loader import (
    Dataset_CrossVar,
    Dataset_FCR_CrossVar,
    Dataset_FCR_RelationPretrain,
)
from data_provider.relation_pretrain_loader import Dataset_RelationPretrain


CROSSVAR_DATASETS = ('fcr_crossvar', 'camels_crossvar', 'ngr_ms_crossvar', 'pwl_ms_crossvar')


def _build_dataset(args, flag):
    common = dict(
        root_path=args.root_path, x_data_path=args.x_data_path,
        y_data_path=args.y_data_path, flag=flag,
        seq_len=args.seq_len, support_ratio=args.support_ratio,
        window_stride=getattr(args, 'window_stride', 1),
        query_ratio=args.query_ratio, encoder_proto_path=args.encoder_proto_path,
        adaptation_validation=flag == 'val' and getattr(args, 'eval_split', 'test') == 'val',
    )
    if args.data == 'fcr_crossvar':
        return Dataset_FCR_CrossVar(**common)
    if args.data not in CROSSVAR_DATASETS:
        raise ValueError(f'Unknown cross-variable dataset: {args.data}')
    return Dataset_CrossVar(
        **common, site_meta_path=getattr(args, 'site_meta_path', None),
        random_site_split=args.data == 'camels_crossvar',
        cluster_result_path=args.cluster_result_path,
        fold_id=args.fold_id,
        verbose=getattr(args, 'verbose', True),
    )


def infer_static_dim(args):
    """Infer context dimension from a training sample, including prototype features."""
    old_verbose = getattr(args, 'verbose', True)
    args.verbose = False
    try:
        data_set = _build_dataset(args, 'train')
    finally:
        args.verbose = old_verbose

    if len(data_set) == 0:
        raise ValueError("Cannot infer static_dim from an empty train dataset.")

    sample = data_set[0]
    static_feat = sample[8]
    return int(static_feat.shape[-1])


def _build_weighted_sampler(data_set):
    """Balance training sites by cluster in single-process runs."""
    cluster_map = data_set.site_cluster_map

    if len(cluster_map) == 0:
        return None

    cluster_counts = collections.Counter(cluster_map.values())

    weights = [
        1.0 / cluster_counts[cluster_map[i]]
        for i in range(len(data_set))
    ]

    weights = torch.DoubleTensor(weights)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(data_set),
        replacement=True
    )


def data_provider(args, flag):
    """Return the dataset and its data loader."""
    if getattr(args, 'is_relation_pretraining', False):
        if flag != 'train':
            raise ValueError("Relation pretraining only builds the train split.")
        Data = Dataset_FCR_RelationPretrain if args.data == 'fcr_crossvar' else Dataset_RelationPretrain
        data_set = Data(
            root_path=args.root_path,
            x_data_path=args.x_data_path,
            y_data_path=args.y_data_path,
            seq_len=args.seq_len,
            samples_per_site=getattr(args, 'relation_pretrain_samples_per_site', 8),
            query_ratio=getattr(args, 'query_ratio', 0.34),
            site_meta_path=getattr(args, 'site_meta_path', None),
            verbose=getattr(args, 'verbose', True),
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        return data_set, data_loader


    data_set = _build_dataset(args, flag)

    if flag == 'test' or (flag == 'val' and getattr(args, 'eval_split', 'test') == 'val'):
        batch_size = 1
        shuffle_flag = False
    else:
        batch_size = args.batch_size
        shuffle_flag = True

    sampler = None

    if flag == 'train':
        sampler = _build_weighted_sampler(data_set)
        if sampler is not None:
            shuffle_flag = False

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )


    return data_set, data_loader
