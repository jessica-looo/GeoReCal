import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _as_3d(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.ndim == 2:
        pred = pred[..., None]
        true = true[..., None]

    return pred, true


def _nanmean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)


def pearson_corr(pred, true, eps=1e-8):
    pred, true = _as_3d(pred, true)

    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    true_centered = true - true.mean(axis=1, keepdims=True)

    numerator = (pred_centered * true_centered).sum(axis=1)
    denominator = np.sqrt((pred_centered ** 2).sum(axis=1) * (true_centered ** 2).sum(axis=1))

    corr = numerator / (denominator + eps)
    corr = np.where(denominator > eps, corr, np.nan)
    return _nanmean_or_nan(corr)


def metric(pred, true):
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), pearson_corr(pred, true)


def result_metrics(pred, true):
    return dict(zip(('mae', 'mse', 'rmse', 'corr'), metric(pred, true)))


def prediction_result_fields(preds, trues, inverse_transform, linear_preds=None):
    """Build NPZ fields and TXT blocks; arrays have one target channel.

    inverse_transform is the dataset's existing 2-D target inverse transform.
    """
    if preds.shape != trues.shape or preds.size == 0:
        raise ValueError('Predictions and truths must have the same nonempty shape')
    if linear_preds is not None and linear_preds.shape != preds.shape:
        raise ValueError('Linear predictions must align with predictions and truths')

    def original(array):
        return np.asarray(inverse_transform(array.astype(np.float64).reshape(-1, 1))).reshape(array.shape)

    fields = {'preds': preds, 'trues': trues,
              'preds_original': original(preds), 'trues_original': original(trues)}
    groups = [('prediction_normalized', '', '', preds, trues),
              ('prediction_original', '', '_original', fields['preds_original'], fields['trues_original'])]
    if linear_preds is not None:
        fields.update(linear_preds=linear_preds, linear_preds_original=original(linear_preds))
        groups += [('linear_normalized', 'linear_', '', linear_preds, trues),
                   ('linear_original', 'linear_', '_original', fields['linear_preds_original'], fields['trues_original'])]
    blocks = []
    for title, prefix, suffix, pred, true in groups:
        values = result_metrics(pred, true)
        fields.update({prefix + key + suffix: value for key, value in values.items()})
        blocks.append('[' + title + ']\n' + '\n'.join(
            f'{key}: {value:.10g}' for key, value in values.items()))
    return fields, '\n\n'.join(blocks)


def save_prediction_results(folder, setting, fields, metrics_text):
    """Write temporary outputs, then replace the two result files."""
    import os
    import tempfile
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.results-', suffix='.npz', delete=False) as f:
            temporary.append(Path(f.name))
            np.savez(f, **fields)
            f.flush()
            os.fsync(f.fileno())
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                         prefix='.metrics-', suffix='.txt', delete=False) as f:
            temporary.append(Path(f.name))
            f.write(setting + '\n\n' + metrics_text + '\n')
            f.flush()
            os.fsync(f.fileno())
        for source, name in zip(temporary, ('results.npz', 'metrics.txt')):
            destination = folder / name
            if destination.exists():
                os.chmod(source, destination.stat().st_mode & 0o777)
            else:
                os.chmod(source, 0o644)
            os.replace(source, destination)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
