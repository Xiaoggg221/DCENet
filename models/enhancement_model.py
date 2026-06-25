import torch
import torch.nn as nn
import torch.nn.functional as F  
import logging
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.archs.segment.hrseg_model import create_hrnet

from models.loss import (
    CharbonnierLoss,
    VGGLoss,
    SSIM,
    ContrastLoss,
    CLIPLOSS,
    HighFrequencyMaskedLoss,
    GradientLoss,
)

logger = logging.getLogger('base')


class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        
        self.prior_warmup_steps = int(train_opt.get('prior_warmup_steps', 5000))
        self.grad_clip_max_norm = float(train_opt.get('grad_clip_max_norm', 1.0))
    
        self.weight_moe = float(train_opt.get('weight_moe', 0.006))
        self.moe_sparsity_scale = float(train_opt.get('moe_sparsity_scale', 0.35))
        self.num_moe_stages = int(train_opt.get('num_moe_stages', 5))
       
        self.hf_ramp_steps = int(train_opt.get('hf_ramp_steps', 12000))
   
        self.ssim_weight = float(train_opt.get('ssim_weight', 0.28))
        self.contrast_weight = float(train_opt.get('contrast_weight', 0.008))
        self.clip_weight = float(train_opt.get('clip_weight', 0.008))
        
   
        self.color_weight = float(train_opt.get('color_weight', 0.5))

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        ####  segment
        if opt['seg']:
            print(" ******************** load segment model *********************")
            self.seg_model = create_hrnet().cuda()
            self.seg_model.eval()
        else:
            self.seg_model = None

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']

            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

            self.is_vgg_loss = train_opt['vgg_loss']
            self.l_pix_w = train_opt['pixel_weight']
            self.cri_pix_ill = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)
            self.con_loss = ContrastLoss().to(self.device)
            self.cri_vgg = VGGLoss().to(self.device)
            self.ssim_loss = SSIM().to(self.device)
            self.l1_loss = torch.nn.L1Loss().to(self.device)
            self.clip_loss = CLIPLOSS().to(self.device)

     
            self.gradloss = GradientLoss().to(self.device)

            
            self.weight_hf = float(train_opt.get('weight_hf', 0.006))
            hf_thr = train_opt.get('hf_threshold', 0.5)
            self.cri_hf = HighFrequencyMaskedLoss(threshold=hf_thr).to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
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

    def combine_elements(self, arr1, arr2):
        combine_dict = {}
        for i in range(len(arr1)):
            combine_dict[i] = {arr1[i], arr2[i]}
        combine_list = list(combine_dict.values())
        return combine_list

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.neg_H = data['NEG'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        if self.seg_model is not None:
            self.seg_map, self.seg_feature = self.seg_model(self.real_H)
        else:
            self.seg_map, self.seg_feature = None, None

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        for name, param in self.netG.named_parameters():
            if step < self.prior_warmup_steps:
                param.requires_grad = "gammanet" in name or "smgm" in name
            else:
                param.requires_grad = not ("gammanet" in name or "smgm" in name)

        self.optimizer_G.zero_grad()

        try:
            forward_out = self.netG(self.var_L, gt=self.real_H)
        except TypeError:
            forward_out = self.netG(self.var_L)

        if len(forward_out) == 3:
            self.fake_H, moe_losses, smgm_dict = forward_out
        else:
            self.fake_H, moe_losses = forward_out
            smgm_dict = None

        l_final = 0.0

        if step < self.prior_warmup_steps:
            if smgm_dict is not None and 'pred_grad' in smgm_dict and 'gt_grad' in smgm_dict:
                l_grad = self.gradloss(smgm_dict['pred_grad'], smgm_dict['gt_grad'])
                l_final += l_grad
                self.log_dict['l_grad'] = l_grad.item()
            elif smgm_dict is not None and 'smgm_loss' in smgm_dict:
                l_final += smgm_dict['smgm_loss']
                self.log_dict['l_grad'] = smgm_dict['smgm_loss'].item()
            else:
                l_final += (self.fake_H.sum() * 0.0)
                self.log_dict['l_grad'] = 0.0
        else:
            _, _, H, W = self.real_H.shape
            c_loss = self.con_loss(self.var_L, self.real_H, self.neg_H, self.fake_H) * self.contrast_weight
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            l_ssim = (1 - self.ssim_loss(self.fake_H, self.real_H)) * self.ssim_weight
            
          
            l_color = 1.0 - F.cosine_similarity(self.fake_H, self.real_H, dim=1).mean()
            l_color_weighted = l_color * self.color_weight

            vgg_loss_state = False
            if self.is_vgg_loss:
                l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.2
                vgg_loss_state = True

            l_hf = 0.0
            w_hf_eff = 0.0
            if self.weight_hf > 0:
                if self.hf_ramp_steps > 0:
                    prog = (step - self.prior_warmup_steps) / float(self.hf_ramp_steps)
                    w_hf_eff = self.weight_hf * min(1.0, max(0.0, prog))
                else:
                    w_hf_eff = self.weight_hf
                if w_hf_eff > 0:
                    l_hf = self.cri_hf(self.fake_H, self.real_H) * w_hf_eff

            l_moe_raw = moe_losses['moe_balance'] + self.moe_sparsity_scale * moe_losses['moe_sparsity']
            l_moe = l_moe_raw / max(1, self.num_moe_stages)

            clip_loss_state = False
            if self.seg_map is not None and step % 200 == 0:
                l_clip = self.clip_loss(self.seg_map, self.fake_H, self.real_H) * self.clip_weight
                if vgg_loss_state:
                    l_final = l_pix + l_ssim + l_clip + c_loss + l_vgg + l_hf + self.weight_moe * l_moe + l_color_weighted
                else:
                    l_final = l_pix + l_ssim + l_clip + c_loss + l_hf + self.weight_moe * l_moe + l_color_weighted
                clip_loss_state = True
                self.log_dict['l_clip'] = l_clip.item()
            else:
                if vgg_loss_state:
                    l_final = l_pix + l_ssim + c_loss + l_vgg + l_hf + self.weight_moe * l_moe + l_color_weighted
                else:
                    l_final = l_pix + l_ssim + c_loss + l_hf + self.weight_moe * l_moe + l_color_weighted

            self.log_dict['c_loss'] = c_loss.item()
            self.log_dict['l_pix'] = l_pix.item()
            self.log_dict['l_ssim'] = l_ssim.item()
            self.log_dict['l_color'] = l_color.item() # 🌟 将 color loss 记录到日志，方便你观察它的下降
            if vgg_loss_state:
                self.log_dict['l_vgg'] = l_vgg.item()
            if self.weight_hf > 0:
                self.log_dict['l_hf'] = l_hf.item() if isinstance(l_hf, torch.Tensor) else l_hf
                self.log_dict['w_hf_eff'] = float(w_hf_eff)
            self.log_dict['l_moe'] = l_moe.item()

        l_final.backward()

        if self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.grad_clip_max_norm)
        self.optimizer_G.step()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            forward_out = self.netG(self.var_L)
  
            if isinstance(forward_out, (tuple, list)):
                self.fake_H = forward_out[0]
            else:
                self.fake_H = forward_out
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)