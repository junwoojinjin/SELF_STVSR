import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss

logger = logging.getLogger('base')


class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)

        self.real_H = ''
        self.fake_H = ''

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
            else:
                raise NotImplementedError(
                    'Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            self.l1_loss = nn.L1Loss(reduction='sum').to(self.device)
            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            optim_params_spy = []
            optim_params_inter = []
            for k, v in self.netG.named_parameters():
                if 'module.fgDCN_forward.' in k or 'module.fgDCN_backward.' in k or 'module.fusion_inter.' in k or 'module.fgDCN_back_feature.' in k or 'module.fgDCM_for_feature.' in k:
                    optim_params_inter.append(v)

                elif 'module.spynet' in k:
                    optim_params_spy.append(v)
                elif v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            'Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizer_spynet = torch.optim.Adam(optim_params_spy, lr=train_opt['lr_G'] * 0.125,
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))

            self.optimizer_inter = torch.optim.Adam(optim_params_inter, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))


            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_spynet)
            self.optimizers.append(self.optimizer_inter)


            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def evaluate_output(self, data):
        self.netG.eval()
        with torch.no_grad():
            self.var_L = data['LQs'].to(self.device)
            self.fake_H, _,_,_ = self.netG(self.var_L)

        self.netG.train()
        return self.fake_H.data.float().cpu().squeeze(0)

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        for i in range(len(self.optimizers)):
            self.optimizers[i].param_groups[0]['lr'] = 0

    def init_logs(self):
        self.log_dict['l_pix'] = 0.
        self.log_dict['l_pix_even'] = 0.
        self.log_dict['l_pix_odd'] = 0.
        self.log_dict['l_pix_inter'] = 0.  # * 0.2
        self.log_dict['l_percep_inter'] = 0.

    def optimize_parameters(self, step): # real feed forward and backprop
        # Base loss
        for i in range(len(self.optimizers)):
            self.optimizers[i].zero_grad()

        N = self.real_H.size()[1]

        self.fake_H, self.fake_H_L, self.long_L1_fea ,self.long_inter_fea, = self.netG(self.var_L)
        #l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        self.fake_even = self.fake_H[:, ::2, :, :, :]
        self.fake_odd = self.fake_H[:, 1::2, :, : ,:]
        self.real_H_even = self.real_H[:, 0::2, :, :, :]
        self.real_H_odd = self.real_H[:, 1::2, :, :, :]

        l_pix_even = self.l_pix_w * self.cri_pix( self.fake_even, self.real_H_even)
        l_pix_odd = self.l_pix_w * self.cri_pix(self.fake_odd, self.real_H_odd)
        l_pix = l_pix_even + l_pix_odd
        self.real_H_in = self.real_H[:, 2: N - 2:2, :, :, :]
        #self.real_H_in = torch.stack([ self.real_H[:, i, :, :, :] for i in range(2, self.real_H.shape[1] - 2 , 2) ],dim=1)
        l_pix2 = self.l_pix_w * self.cri_pix(self.fake_H_L , self.real_H_in)
        l_inter = self.l1_loss(self.long_inter_fea , self.long_L1_fea)

        #loss_all = l_pix + l_pix2  + l_inter
        loss_all =  l_pix + l_pix2 * 0.2 #+ l_inter * 0.1

        loss_all.backward()

        for i in range(0 , len(self.optimizers)):
            self.optimizers[i].step()

        # set log
        self.log_dict['l_pix'] += l_pix.item()
        self.log_dict['l_pix_even'] += l_pix_even.item()
        self.log_dict['l_pix_odd'] += l_pix_odd.item()
        self.log_dict['l_pix_inter'] += l_pix2.item() #* 0.2
        self.log_dict['l_percep_inter'] += l_inter.item() #* 0.2


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['STVSR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['INT_VSR'] = self.fake_H_L.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
