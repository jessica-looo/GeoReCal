import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import GeoReCal_relation_pretrain


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class Exp_Relation_Pretrain(Exp_Basic):
    def _build_model(self):
        self.args.static_dim = 3
        return GeoReCal_relation_pretrain.Model(self.args)

    def _get_data(self, flag='train'):
        return data_provider(self.args, flag)

    def _augment(self, x):
        out = x
        noise_std = float(getattr(self.args, 'relation_aug_noise_std', 0.05))
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std

        mask_ratio = float(getattr(self.args, 'relation_aug_mask_ratio', 0.1))
        if mask_ratio > 0:
            mask = (torch.rand(out.shape[0], out.shape[1], 1, device=out.device) > mask_ratio).float()
            out = out * mask
        return out

    def _masked_infonce(self, q, k, site_id_q=None, site_id_k=None, temperature=0.2):
        logits = torch.matmul(q, k.T) / temperature
        bsz = q.shape[0]
        labels = torch.arange(bsz, device=q.device)

        if site_id_q is not None and site_id_k is not None:
            same_site = site_id_q.view(-1, 1) == site_id_k.view(1, -1)
            eye = torch.eye(bsz, dtype=torch.bool, device=q.device)

            # Keep the diagonal positive; other same-site samples are not negatives.
            logits = logits.masked_fill(same_site & (~eye), -1e9)

        return F.cross_entropy(logits, labels)

    def _update_memory(self, memory, seen, site_ids, reps, momentum):
        with torch.no_grad():
            for sid in site_ids.unique():
                mask = site_ids == sid
                cur = F.normalize(reps[mask].mean(dim=0), dim=0)
                sid_int = int(sid.item())
                if not bool(seen[sid_int]):
                    memory[sid_int] = cur
                    seen[sid_int] = True
                else:
                    memory[sid_int] = F.normalize(momentum * memory[sid_int] + (1.0 - momentum) * cur, dim=0)

    def _cluster_memory(self, memory, seen):
        valid_mask = seen.detach().cpu().numpy().astype(bool)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return None, None, None

        mem_np = memory.detach().cpu().numpy()
        k = min(int(getattr(self.args, 'num_prototypes', 6)), len(valid_idx))
        if k <= 0:
            return None, None, None

        kmeans = KMeans(n_clusters=k, random_state=2021, n_init=10)
        labels_valid = kmeans.fit_predict(mem_np[valid_idx])

        labels = np.full(memory.shape[0], -1, dtype=np.int64)
        labels[valid_idx] = labels_valid
        prototypes = kmeans.cluster_centers_.astype(np.float32)
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        site_sim = mem_np @ prototypes.T
        return torch.tensor(prototypes, dtype=torch.float32, device=memory.device), labels, site_sim.astype(np.float32)

    def _save_outputs(self, setting, data_set, memory, seen, prototypes_t, labels, site_sim):
        path = self.args.checkpoints
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        with open(os.path.join(path, 'checkpoint_config.json'), 'w') as f:
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

        if prototypes_t is None or labels is None or site_sim is None:
            prototypes_t, labels, site_sim = self._cluster_memory(memory, seen)

        if prototypes_t is None:
            raise RuntimeError("No relation prototypes were produced; memory bank is empty.")

        prototypes = prototypes_t.detach().cpu().numpy().astype(np.float32)
        site_names = np.asarray(data_set.site_names)
        site_memory = memory.detach().cpu().numpy().astype(np.float32)

        cluster_members = defaultdict(list)
        for idx, label in enumerate(labels):
            if label >= 0:
                cluster_members[int(label)].append(str(site_names[idx]))

        metadata = {
            'prototype_mode': 'relation_pretrain_online_kmeans',
            'num_prototypes': int(prototypes.shape[0]),
            'relation_components': ['zx', 'zy', 'zx_times_zy'],
            'relation_mode': 'joint',
            'fcr_pretrain_scope': 'all',
            'pretrain_sites': list(map(str, data_set.site_names)),
            'cluster_members': dict(cluster_members),
        }

        output_path = os.path.join(path, 'prototypes.npz')

        np.savez(
            output_path,
            site_names=site_names,
            site_cluster_label=np.asarray(labels, dtype=np.int64),
            prototypes=prototypes,
            site_memory=site_memory,
            site_sim=np.asarray(site_sim, dtype=np.float32),
            cluster_members_json=json.dumps(dict(cluster_members), ensure_ascii=False),
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        print(f"Pretraining saved: {path} (checkpoint.pth, prototypes.npz)")

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        num_sites = train_data.num_sites
        memory = torch.zeros(num_sites, self.args.d_model, device=self.device)
        seen = torch.zeros(num_sites, dtype=torch.bool, device=self.device)
        prototypes_t = None
        labels = None
        site_sim = None

        site_w = float(getattr(self.args, 'relation_loss_site_weight', 1.0))
        aug_w = float(getattr(self.args, 'relation_loss_aug_weight', 0.5))
        xy_w = float(getattr(self.args, 'relation_loss_xy_weight', 0.1))
        proto_w = float(getattr(self.args, 'relation_loss_proto_weight', 0.2))
        proto_temp = float(getattr(self.args, 'prototype_temperature', 0.2))
        memory_m = float(getattr(self.args, 'prototype_momentum', 0.9))
        warmup_epochs = int(getattr(self.args, 'prototype_warmup_epochs', 5))
        cluster_interval = max(1, int(getattr(self.args, 'prototype_cluster_interval', 2)))
        early_stop_epoch = int(getattr(self.args, 'relation_early_stop_epoch', 0))
        use_early_stop = early_stop_epoch > 0
        best_loss = float('inf')
        bad_epochs = 0
        best_snapshot = None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            losses = []
            t0 = time.time()

            if epoch >= warmup_epochs and (epoch - warmup_epochs) % cluster_interval == 0:
                prototypes_t, labels, site_sim = self._cluster_memory(memory, seen)

            for batch in train_loader:
                x1, y1, _m1, x2, y2, _m2, static, site_ids = [item.to(self.device) for item in batch]

                zx1, zy1, r1 = self.model(x1, y1, static)
                _zx2, _zy2, r2 = self.model(x2, y2, static)
                x_aug = self._augment(x1)
                y_aug = self._augment(y1)
                _, _, r1_aug = self.model(x_aug, y_aug, static)

                loss_site = self._masked_infonce(r1, r2, site_ids, site_ids)
                loss_aug = self._masked_infonce(r1, r1_aug, site_ids, site_ids)
                loss_xy = self._masked_infonce(zx1, zy1, site_ids, site_ids)

                loss_proto = torch.tensor(0.0, device=self.device)
                if prototypes_t is not None and labels is not None:
                    target_np = labels[site_ids.detach().cpu().numpy()]
                    valid = target_np >= 0
                    if np.any(valid):
                        target = torch.tensor(target_np[valid], dtype=torch.long, device=self.device)
                        logits = torch.matmul(r1[valid], prototypes_t.T) / proto_temp
                        loss_proto = F.cross_entropy(logits, target)

                loss = site_w * loss_site + aug_w * loss_aug + xy_w * loss_xy + proto_w * loss_proto

                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite relation pretraining loss')
                loss.backward()
                optimizer.step()

                self._update_memory(memory, seen, site_ids, r1.detach(), memory_m)
                losses.append(float(loss.detach().cpu()))

            avg_loss = float(np.mean(losses)) if losses else float('nan')
            print(f"Pretrain epoch {epoch + 1}/{self.args.train_epochs}: loss={avg_loss:.6f} time={time.time() - t0:.1f}s")

            if use_early_stop and epoch >= warmup_epochs and np.isfinite(avg_loss):
                improved = avg_loss < best_loss
                if improved:
                    best_loss = avg_loss
                    bad_epochs = 0
                    snapshot_prototypes_t, snapshot_labels, snapshot_site_sim = self._cluster_memory(memory, seen)
                    best_snapshot = {
                        'epoch': epoch + 1,
                        'model': {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        },
                        'memory': memory.detach().clone(),
                        'seen': seen.detach().clone(),
                        'prototypes_t': (
                            snapshot_prototypes_t.detach().clone()
                            if snapshot_prototypes_t is not None else None
                        ),
                        'labels': snapshot_labels.copy() if snapshot_labels is not None else None,
                        'site_sim': snapshot_site_sim.copy() if snapshot_site_sim is not None else None,
                    }
                else:
                    bad_epochs += 1

                if bad_epochs > 0:
                    print(f"Early stopping: {bad_epochs}/{early_stop_epoch}")

                if bad_epochs >= early_stop_epoch:
                    print(
                        f"Early stopping relation pretrain at epoch {epoch + 1}. "
                        f"Best epoch: {best_snapshot['epoch'] if best_snapshot else 'N/A'}, "
                        f"best train loss: {best_loss:.12f}"
                    )
                    break

        if use_early_stop and best_snapshot is not None:
            self.model.load_state_dict(best_snapshot['model'], strict=True)
            memory = best_snapshot['memory'].to(self.device)
            seen = best_snapshot['seen'].to(self.device)
            prototypes_t = (
                best_snapshot['prototypes_t'].to(self.device)
                if best_snapshot['prototypes_t'] is not None else None
            )
            labels = best_snapshot['labels']
            site_sim = best_snapshot['site_sim']
            print(
                f"Restored best relation-pretrain snapshot from epoch "
                f"{best_snapshot['epoch']} with train loss {best_loss:.12f}"
            )

        prototypes_t, labels, site_sim = self._cluster_memory(memory, seen)
        self._save_outputs(setting, train_data, memory, seen, prototypes_t, labels, site_sim)
        if use_early_stop and best_snapshot is not None:
            with open(os.path.join(self.args.checkpoints, 'selection.json'), 'w') as f:
                json.dump({'metric': 'pretrain_loss', 'value': best_loss, 'epoch': best_snapshot['epoch']}, f)
        return self.model
