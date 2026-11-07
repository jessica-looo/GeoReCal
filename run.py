import argparse
import os
import torch
from exp.maml_exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_relation_pretrain import Exp_Relation_Pretrain
import random
import numpy as np

def configure_experiment(args, stage, iteration=0):
    output_root = args.experiment_dir
    result_stage = 'maml_baseline' if stage == 'maml' else stage
    args.checkpoints = os.path.join(output_root, result_stage)
    if stage == 'maml' and args.itr > 1:
        args.checkpoints = os.path.join(args.checkpoints, f"run_{iteration}")
    if args.fold_id is not None and stage == 'maml':
        args.checkpoints = os.path.join(args.checkpoints, f"fold_{args.fold_id}")
    os.makedirs(args.checkpoints, exist_ok=True)

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GeoReCal')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='crossvar', help='experiment label')
    parser.add_argument('--cluster_result_path', type=str, default=None, help='site-cluster CSV path')
    parser.add_argument('--encoder_proto_path', type=str, default=None, help='offline encoder prototype npz path')

    parser.add_argument('--data', type=str, required=True, choices=['fcr_crossvar', 'camels_crossvar', 'ngr_ms_crossvar', 'pwl_ms_crossvar'], help='cross-variable dataset')
    parser.add_argument('--site_meta_path', default=None)
    parser.add_argument('--root_path', type=str, required=True, help='root path of the data file')
    parser.add_argument('--x_data_path', type=str, required=True, help='auxiliary variable CSV')
    parser.add_argument('--y_data_path', type=str, required=True, help='target variable CSV')
    parser.add_argument('--experiment_dir', type=str, default=None, help='experiment output directory under results')
    parser.add_argument('--support_ratio', type=float, default=None, help='crossvar test-only support fraction of the full aligned record; None uses all pre-query history')
    parser.add_argument('--query_ratio', type=float, default=0.34, help='crossvar query fraction retained at the end of each aligned record')
    parser.add_argument("--fold_id", type=int, choices=range(5), default=None)

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--patch_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--window_stride', type=int, default=1, help='MAML data-window step; independent of patch stride')
    parser.add_argument('--stride', type=int, default=64, help='stride between patch')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--domain_len', type=int, help='the number of domain', default=128)

    parser.add_argument('--static_emb_dim', type=int, help='static embedding dim', default=8)

    parser.add_argument('--meta_prefix_len', type=int, default = 8, help='meta prefix len')

    parser.add_argument('--is_relation_pretraining', action='store_true', help='run crossvar relation self-supervised pretraining instead of forecasting MAML')
    parser.add_argument('--relation_loss_site_weight', type=float, default=1.0, help='weight for same-site relation contrastive loss')
    parser.add_argument('--relation_loss_aug_weight', type=float, default=0.5, help='weight for same-window augmentation consistency loss')
    parser.add_argument('--relation_loss_xy_weight', type=float, default=0.1, help='weight for same-window x/y alignment loss')
    parser.add_argument('--relation_loss_proto_weight', type=float, default=0.2, help='weight for online prototype assignment loss')
    parser.add_argument('--relation_aug_noise_std', type=float, default=0.05, help='std of Gaussian noise in relation pretrain augmentation')
    parser.add_argument('--relation_aug_mask_ratio', type=float, default=0.1, help='time mask ratio in relation pretrain augmentation')
    parser.add_argument('--relation_pretrain_samples_per_site', type=int, default=8, help='number of sampled window pairs per site per relation-pretrain epoch')
    parser.add_argument('--num_prototypes', type=int, default=6, help='number of KMeans prototypes for relation pretraining')
    parser.add_argument('--prototype_warmup_epochs', type=int, default=5, help='epochs before enabling prototype loss')
    parser.add_argument('--prototype_cluster_interval', type=int, default=2, help='KMeans recluster interval after warmup')
    parser.add_argument('--prototype_momentum', type=float, default=0.9, help='EMA momentum for site memory bank')
    parser.add_argument('--prototype_temperature', type=float, default=0.2, help='temperature for prototype assignment logits')
    parser.add_argument('--relation_pretrain_checkpoint', type=str, default=None, help='checkpoint from relation pretraining; loads backbone.* weights into GeoReCal before MAML')
    parser.add_argument('--relation_early_stop_epoch', type=int, default=0, help='early stop relation pretraining after this many epochs without train-loss improvement; 0 disables it')

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')

    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=True, help='use GPU when available; --no-use_gpu selects CPU')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    parser.add_argument('--maml_loss', choices=['mae', 'mse'], default='mae')
    parser.add_argument('--selection_metric', choices=['mae', 'mse'], default='mae')
    parser.add_argument('--eval_split', choices=['val', 'test'], default='test')
    parser.add_argument('--evaluation_dir', default=None)
    parser.add_argument('--skip_test', action='store_true', help='validation-only tuning; do not evaluate test tasks')
    parser.add_argument('--inner_steps', type=int, default=2, help='number of adaptation steps on support data')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='inner-loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='outer-loop learning rate')

    args = parser.parse_args()
    if args.data not in ('fcr_crossvar', 'camels_crossvar') and not args.is_relation_pretraining and args.fold_id is None:
        parser.error('Site-level downstream experiments require --fold_id 0 through 4')
    if args.data != 'fcr_crossvar' and args.site_meta_path is None:
        parser.error('Site metadata is required: --site_meta_path')
    if args.data in ('ngr_ms_crossvar', 'pwl_ms_crossvar') and not args.is_relation_pretraining and args.cluster_result_path is None:
        parser.error('Sediment downstream experiments require --cluster_result_path')
    args.experiment_dir = args.experiment_dir or os.path.join('results', args.data, args.model_id)
    if not args.is_relation_pretraining:
        args.encoder_proto_path = args.encoder_proto_path or os.path.join(args.experiment_dir, 'pretrain', 'prototypes.npz')
        args.relation_pretrain_checkpoint = args.relation_pretrain_checkpoint or os.path.join(args.experiment_dir, 'pretrain', 'checkpoint.pth')
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.is_relation_pretraining:
        args.setting = f'GeoReCal_{args.data}_relation_sl{args.seq_len}_dm{args.d_model}'
        configure_experiment(args, 'pretrain')
        Exp_Relation_Pretrain(args).train(args.setting)
    else:
        for iteration in range(args.itr):
            args.setting = f'GeoReCal_{args.data}_sl{args.seq_len}_dm{args.d_model}_run{iteration}'
            configure_experiment(args, 'maml', iteration)
            exp = Exp_Long_Term_Forecast(args)
            if args.is_training:
                exp.train(args.setting)
                if not args.skip_test:
                    exp.test(args.setting)
            else:
                exp.test(args.setting, test=1)
            torch.cuda.empty_cache()
