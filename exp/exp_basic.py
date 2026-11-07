import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}; CUDA_VISIBLE_DEVICES={}'.format(self.args.gpu, os.environ.get('CUDA_VISIBLE_DEVICES', 'all')))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
