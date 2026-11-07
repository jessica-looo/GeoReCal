from models import GeoReCal
from data_provider.data_factory import data_provider, infer_static_dim
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.metrics import prediction_result_fields, save_prediction_results
import torch
import torch.nn as nn
from torch.func import functional_call
import os
import time
import warnings
import json
import numpy as np

warnings.filterwarnings('ignore')

def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)

class Exp_Long_Term_Forecast(Exp_Basic):

    def _build_model(self):
        self.args.static_dim = infer_static_dim(self.args)

        self.args.device = self.device
        model = GeoReCal.Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        return nn.MSELoss() if getattr(self.args, 'maml_loss', 'mae') == 'mse' else nn.L1Loss()

    def _forward_query(self, params, t_qx, t_qx_mark, t_qy, t_qy_mark, t_static_q):
        inputs = (t_qx, t_qx_mark, t_qy, t_qy_mark, t_static_q)
        if params is None:
            return self.model(*inputs)
        return functional_call(self.model, params, inputs)

    def _add_linear_baseline(self, outputs, baseline):
        return outputs + baseline

    @torch.no_grad()
    def _site_linear_baseline(self, sx, sy, qx):
        assert sx.ndim == 3 and sx.shape == sy.shape and sx.shape[-1] == 1
        assert qx.shape[1:] == sx.shape[1:]
        assert sx.size(0) > 0
        step = getattr(self.args, 'window_stride', 1)
        assert 1 <= step <= sx.size(1)
        # Recover each observed support point once, rather than weighting overlapping windows.
        x = torch.cat([sx[0], sx[1:, -step:].reshape(-1, 1)], dim=0)
        y = torch.cat([sy[0], sy[1:, -step:].reshape(-1, 1)], dim=0)
        xc, yc = x - x.mean(), y - y.mean()
        a = (xc * yc).sum() / xc.square().sum().clamp_min(1e-12)
        b = y.mean() - a * x.mean()
        return a * sx + b, a * qx + b

    def _predict_windows(self, params, x, x_mark, y, y_mark, static_feat, baseline):
        static = static_feat.unsqueeze(0).repeat(x.size(0), 1)
        outputs = self._forward_query(params, x, x_mark, y, y_mark, static)
        return self._add_linear_baseline(outputs, baseline), y

    def _adapt_and_predict(self, params, sx, sy, sx_mark, sy_mark,
                           qx, qy, qx_mark, qy_mark, static_feat,
                           linear_s, linear_q, criterion, inner_steps, inner_lr,
                           query_grad=False):
        # Each task starts from the shared initialization; clones retain outer gradients.
        with torch.enable_grad():
            fast_weights = {k: v.clone() for k, v in params.items()} if params is not None else None
            for _ in range(inner_steps):
                prediction, target = self._predict_windows(
                    fast_weights, sx, sx_mark, sy, sy_mark, static_feat, linear_s)
                loss = criterion(prediction, target)
                grads = torch.autograd.grad(
                    loss, fast_weights.values(), create_graph=False, allow_unused=True)
                # First-order MAML: stop gradients through the support gradient only.
                fast_weights = {
                    name: param - inner_lr * grad if grad is not None else param.clone()
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }
        with torch.set_grad_enabled(query_grad):
            return self._predict_windows(
                fast_weights, qx, qx_mark, qy, qy_mark, static_feat, linear_q)

    def vali(self, vali_loader, criterion, maml_params_dict=None):
        total_loss = []
        metric = nn.MSELoss() if getattr(self.args, 'selection_metric', getattr(self.args, 'maml_loss', 'mae')) == 'mse' else nn.L1Loss()
        self.model.eval()

        inner_steps = getattr(self.args, 'inner_steps', 2)
        inner_lr = getattr(self.args, 'inner_lr', 0.01)

        for i, task_data in enumerate(vali_loader):
            support_x, support_y, support_x_mark, support_y_mark, \
            query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years = [item.to(self.device) for item in task_data]

            meta_batch_size = support_x.size(0)

            for task_idx in range(meta_batch_size):
                t_sx = support_x[task_idx]
                t_sy = support_y[task_idx]
                t_sx_mark = support_x_mark[task_idx]
                t_sy_mark = support_y_mark[task_idx]

                t_qx = query_x[task_idx]
                t_qy = query_y[task_idx]
                t_qx_mark = query_x_mark[task_idx]
                t_qy_mark = query_y_mark[task_idx]
                linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)

                outputs_query, t_qy_pred = self._adapt_and_predict(
                    maml_params_dict,
                    t_sx, t_sy, t_sx_mark, t_sy_mark,
                    t_qx, t_qy, t_qx_mark, t_qy_mark, static_feat[task_idx],
                    linear_s, linear_q, criterion, inner_steps, inner_lr,
                    query_grad=False)

                meta_loss = metric(outputs_query, t_qy_pred)
                if not torch.isfinite(meta_loss):
                    raise RuntimeError('Nonfinite validation loss')
                total_loss.append(meta_loss.item())

        torch.set_grad_enabled(True)
        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if not getattr(self.args, 'skip_test', False):
            test_data, test_loader = self._get_data(flag='test')

        relation_ckpt_path = getattr(self.args, 'relation_pretrain_checkpoint', None)
        if relation_ckpt_path:
            print(f'Loading relation-pretrained backbone: {relation_ckpt_path}')
            relation_dict = torch.load(relation_ckpt_path, map_location=self.device)
            model_dict = self.model.state_dict()
            filtered_dict = {}
            for k, v in relation_dict.items():
                if k.startswith('backbone.'):
                    k_model = k.replace('backbone.', '', 1)
                elif k.startswith('module.backbone.'):
                    k_model = k.replace('module.backbone.', '', 1)
                else:
                    continue

                if k_model not in model_dict:
                    raise ValueError(f'Unknown pretrained backbone tensor: {k_model}')
                if v.shape != model_dict[k_model].shape:
                    if k_model.startswith(('static_encoder.', 'static_meta_encoder.')):
                        continue
                    raise ValueError(f'Pretrained backbone shape mismatch: {k_model}')
                filtered_dict[k_model] = v

            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict, strict=False)

        if self.args.train_epochs > 0:
            with torch.no_grad():
                for head in self.model.pretrain_heads:
                    head.linear.weight.zero_()
                    head.linear.bias.zero_()

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        config_path = os.path.join(path, 'checkpoint_config.json')
        with open(config_path, 'w') as f:
            json.dump(
                {
                    'format_version': 1,
                    'setting': setting,
                    'args': {k: _json_safe(v) for k, v in vars(self.args).items()},
                },
                f,
                indent=2,
                sort_keys=True,
            )

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        maml_params_dict = self.model.get_maml_parameters()
        meta_optimizer = torch.optim.Adam(maml_params_dict.values(), lr=self.args.meta_lr)
        inner_steps = getattr(self.args, 'inner_steps', 2)
        inner_lr = getattr(self.args, 'inner_lr', 0.01)
        print("MAML training")

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, task_data in enumerate(train_loader): # One meta-batch contains batch_size sites.
                meta_optimizer.zero_grad()

                support_x, support_y, support_x_mark, support_y_mark, \
                query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years = [item.to(self.device) for item in task_data]

                meta_batch_size = support_x.size(0)
                batch_meta_loss = 0.0

                for task_idx in range(meta_batch_size):
                    t_sx = support_x[task_idx]     # [n_support_windows, seq_len, 1]
                    t_sy = support_y[task_idx]
                    t_sx_mark = support_x_mark[task_idx]
                    t_sy_mark = support_y_mark[task_idx]

                    t_qx = query_x[task_idx]       # [n_query_windows, seq_len, 1]
                    t_qy = query_y[task_idx]
                    t_qx_mark = query_x_mark[task_idx]
                    t_qy_mark = query_y_mark[task_idx]
                    linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)

                    outputs_query, t_qy_pred = self._adapt_and_predict(
                        maml_params_dict,
                        t_sx, t_sy, t_sx_mark, t_sy_mark,
                        t_qx, t_qy, t_qx_mark, t_qy_mark, static_feat[task_idx],
                        linear_s, linear_q, criterion, inner_steps, inner_lr,
                        query_grad=True)

                    meta_loss = criterion(outputs_query, t_qy_pred)
                    batch_meta_loss += meta_loss

                batch_meta_loss = batch_meta_loss / meta_batch_size
                train_loss.append(batch_meta_loss.item())

                if not torch.isfinite(batch_meta_loss):
                    raise RuntimeError('Nonfinite training loss')
                batch_meta_loss.backward()
                meta_optimizer.step()

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_loader, criterion, maml_params_dict)
            test_loss = (self.vali(test_loader, criterion, maml_params_dict)
                         if not getattr(self.args, 'skip_test', False) else float('nan'))

            test_log = f" test={test_loss:.6f}" if not self.args.skip_test else ""
            print(f"MAML epoch {epoch + 1}/{self.args.train_epochs}: train={train_loss:.6f} "
                  f"val={vali_loss:.6f}{test_log} time={time.time() - epoch_time:.1f}s")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.counter > 0:
                print(f"Early stopping: {early_stopping.counter}/{early_stopping.patience}")

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}/{self.args.train_epochs}")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        score = self.vali(vali_loader, criterion, self.model.get_maml_parameters())
        with open(os.path.join(path, 'selection.json'), 'w') as f:
            json.dump({'metric': getattr(self.args, 'selection_metric', 'mae'), 'value': float(score)}, f)

        return self.model

    def test(self, setting, test=0):
        split = getattr(self.args, 'eval_split', 'test')
        test_data, test_loader = self._get_data(flag=split)

        if test:
            print(f"Loading checkpoint: {os.path.join(self.args.checkpoints, 'checkpoint.pth')}")
            state = torch.load(os.path.join(self.args.checkpoints, 'checkpoint.pth'), map_location=self.device)
            self.model.load_state_dict(state)

        preds = []
        trues = []
        linear_preds = []
        sample_sites = []
        sample_years = []
        sample_clusters = []

        folder_path = getattr(self.args, 'evaluation_dir', None) or self.args.checkpoints
        if split == 'val':
            folder_path = os.path.join(folder_path, 'validation')
        os.makedirs(folder_path, exist_ok=True)

        if test:
            config_path = os.path.join(folder_path, 'checkpoint_config.json')
            config = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            config['evaluation'] = {
                'checkpoint': os.path.abspath(os.path.join(self.args.checkpoints, 'checkpoint.pth')),
                'setting': setting, 'args': {k: _json_safe(v) for k, v in vars(self.args).items()}}
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=True)

        self.model.eval()
        maml_params_dict = self.model.get_maml_parameters()
        inner_steps = getattr(self.args, 'inner_steps', 2)
        inner_lr = getattr(self.args, 'inner_lr', 0.01)
        criterion = self._select_criterion()

        global_task_idx = 0

        for i, task_data in enumerate(test_loader):
            support_x, support_y, support_x_mark, support_y_mark, \
            query_x, query_y, query_x_mark, query_y_mark, static_feat, query_years = [item.to(self.device) for item in task_data]

            meta_batch_size = support_x.size(0)

            for task_idx in range(meta_batch_size):
                site_idx = global_task_idx
                site_name = test_data.current_sites[site_idx]
                global_task_idx += 1

                cluster_id = test_data.site_cluster_map.get(site_idx, -1)

                t_qy_years = query_years[task_idx].detach().cpu().numpy()

                t_sx = support_x[task_idx]
                t_sy = support_y[task_idx]
                t_sx_mark = support_x_mark[task_idx]
                t_sy_mark = support_y_mark[task_idx]

                t_qx = query_x[task_idx]
                t_qy = query_y[task_idx]
                t_qx_mark = query_x_mark[task_idx]
                t_qy_mark = query_y_mark[task_idx]
                linear_s, linear_q = self._site_linear_baseline(t_sx, t_sy, t_qx)

                outputs_query, t_qy_pred = self._adapt_and_predict(
                    maml_params_dict,
                    t_sx, t_sy, t_sx_mark, t_sy_mark,
                    t_qx, t_qy, t_qx_mark, t_qy_mark, static_feat[task_idx],
                    linear_s, linear_q, criterion, inner_steps, inner_lr,
                    query_grad=False)

                pred = outputs_query.detach().cpu().numpy()
                true = t_qy_pred.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                if torch.is_tensor(linear_q):
                    linear_preds.append(linear_q.detach().cpu().numpy())

                n_windows = pred.shape[0]
                sample_sites.extend([site_name] * n_windows)
                sample_clusters.extend([cluster_id] * n_windows)
                sample_years.extend(t_qy_years)

        torch.set_grad_enabled(True)

        preds = np.concatenate(preds, axis=0) # Pool query windows across sites.
        trues = np.concatenate(trues, axis=0)
        linear_preds = np.concatenate(linear_preds, axis=0) if linear_preds else None
        sample_sites = np.array(sample_sites)
        sample_clusters = np.array(sample_clusters)
        sample_years = np.array(sample_years)

        full_x_series = []
        full_y_series = []
        full_years = []
        full_sites = []
        full_support_lens = []
        for site_idx, site_name in enumerate(test_data.current_sites):
            full_sites.append(site_name)
            full_x_series.append(np.asarray(test_data.data_x_list[site_idx], dtype=np.float32))
            full_y_series.append(np.asarray(test_data.data_y_list[site_idx], dtype=np.float32))
            full_years.append(np.asarray(test_data.year_list[site_idx], dtype=np.float32))
            full_support_lens.append(int(test_data.support_len_list[site_idx]))

        extra_npz_fields = {
            'crossvar_full_sites': np.asarray(full_sites),
            'crossvar_full_x': np.asarray(full_x_series, dtype=object),
            'crossvar_full_y': np.asarray(full_y_series, dtype=object),
            'crossvar_full_years': np.asarray(full_years, dtype=object),
            'crossvar_full_support_lens': np.asarray(full_support_lens, dtype=np.int64),
            'crossvar_x_path': np.asarray(getattr(self.args, 'x_data_path', '')),
            'crossvar_y_path': np.asarray(getattr(self.args, 'y_data_path', '')),
            'query_ratio': np.asarray(self.args.query_ratio),
            'support_ratio': np.asarray(self.args.support_ratio if self.args.support_ratio is not None else np.nan),
        }


        if not np.isfinite(preds).all() or not np.isfinite(trues).all():
            raise RuntimeError('Nonfinite test predictions or targets')
        fields, metrics_text = prediction_result_fields(
            preds, trues, test_data.inverse_transform, linear_preds)
        fields.update(sites=sample_sites, clusters=sample_clusters, years=sample_years,
                      **extra_npz_fields)
        scalers = {'x': test_data.x_scaler, 'y': test_data.y_scaler}
        for name, scaler in scalers.items():
            if scaler is not None:
                fields[name + '_scaler_mean'] = scaler.mean_.copy()
                fields[name + '_scaler_scale'] = scaler.scale_.copy()
        print(f"Evaluation: mse={fields['mse']:.6f} mae={fields['mae']:.6f}")
        save_prediction_results(folder_path, setting, fields, metrics_text)
        print(f"Results saved: {os.path.join(folder_path, 'results.npz')}")

        return
