import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.arch_util import *
from models.archs.SS2D_arch import SS2D6
from models.archs.ffc import *
from models.archs.wtconv.wtconv2d import *

from models.loss import GradientLoss 

logger = logging.getLogger('base')


class GradientPrior(nn.Module):
    def __init__(self):
        super(GradientPrior, self).__init__()
      
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image):
        b, c, h, w = image.shape
       
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_x = F.conv2d(image, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=c)

        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient


class GammaNet(nn.Module):
    def __init__(self, input_channels=3, feature_channels=16):
        super(GammaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels * 2, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        gamma = self.gamma_pred(features)
        gamma = gamma.view(-1, 1, 1, 1)
        return gamma

class MCSM(nn.Module):
    def __init__(self, input_channels=3, feature_channels=16):
        super(MCSM, self).__init__()
        self.gammanet = GammaNet(input_channels=input_channels, feature_channels=feature_channels)
        self.grad_extractor = GradientPrior() # 🌟 使用封装好的安全算子

    def forward(self, x_LL):
        c_min = x_LL.amin(dim=(2, 3), keepdim=True)
        c_max = x_LL.amax(dim=(2, 3), keepdim=True)
        normalized_x_LL = (x_LL - c_min) / (c_max - c_min + 1e-8)
        
        gamma = self.gammanet(normalized_x_LL)
        gamma = torch.clamp(gamma, min=0.1, max=3.0)
        
        S_i = torch.pow(normalized_x_LL + 1e-8, gamma)
        S_i_denorm = S_i * (c_max - c_min + 1e-8) + c_min
        
        G_i = self.grad_extractor(S_i_denorm) # 🌟 统一调用
        return gamma, S_i_denorm, G_i



class Expert_Base(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
    def forward(self, x):
        return self.block(x)

class Expert_LargeKernel(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch),
            nn.GroupNorm(1, ch),  
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class Expert_Attention(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        reduced_ch = max(1, ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, reduced_ch, 1),
            nn.GELU(),
            nn.Conv2d(reduced_ch, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        attn_weight = self.se(out)
        return out * attn_weight

class Expert_MultiScale(nn.Module):
    def __init__(self, ch):
        super().__init__()
        half_ch = ch // 2
        self.branch1 = nn.Conv2d(ch, half_ch, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(ch, half_ch, kernel_size=3, padding=2, dilation=2)
        self.act = nn.GELU()
        self.fuse = nn.Conv2d(half_ch * 2, ch, kernel_size=1)
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        return self.fuse(self.act(out))

class AttentionRouter(nn.Module):
    def __init__(self, ch, n_expert):
        super().__init__()
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid()
        )
        self.pixel_scorer = nn.Conv2d(ch, n_expert, 1)

    def forward(self, x):
        ca_feat = x * self.global_context(x)
        score = self.pixel_scorer(ca_feat)  
        return score

class HeteroSpatialMoE(nn.Module):
    def __init__(self, ch, n_expert=4, topk=2):
        super().__init__()
        self.topk = topk
        self.experts = nn.ModuleList([
            Expert_Base(ch),         
            Expert_LargeKernel(ch),  
            Expert_Attention(ch),    
            Expert_MultiScale(ch)    
        ])
        self.router = AttentionRouter(ch, n_expert)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        score = self.router(feat)
        topk_val, topk_idx = torch.topk(score, k=self.topk, dim=1)
        gate = torch.zeros_like(score).scatter(1, topk_idx, topk_val)
        gate = torch.softmax(gate, dim=1)  
        
        out_moe = 0
        for e_idx, expert in enumerate(self.experts):
            w = gate[:, e_idx:e_idx + 1, :, :]
            out_moe = out_moe + w * expert(feat)

        out = feat + self.gamma * out_moe

        p = gate.mean(dim=(0, 2, 3))
        balance_loss = torch.var(p)
        moe_sparsity = (gate ** 2).mean()

        return out, {'moe_balance': balance_loss, 'moe_sparsity': moe_sparsity}




class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        
        y, var, weight = ctx.saved_tensors 
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LightBlock(nn.Module):
    def __init__(self, dim):
        super(LightBlock, self).__init__()
        self.channel = dim
        self.SIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, bias=True),
        )
        self.CIM = nn.Sequential(
            LayerNorm2d(dim),
            FFCResnetBlock(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True),
        )
        self.moe_ln = LayerNorm2d(dim)
        self.moe_conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, bias=True)
        self.moe_sg = SimpleGate()
        self.moe_module = HeteroSpatialMoE(dim)
        self.moe_conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=False)
        
        with torch.no_grad():
            self.fusion_conv.weight.fill_(0.0)
            for i in range(dim):
                self.fusion_conv.weight[i, dim + i, 0, 0] = 0.1
        nn.init.constant_(self.moe_module.gamma, 0.1)

    def forward(self, x):
        y_lfeb = self.SIM(x) + x
        y_lfeb = self.CIM(y_lfeb) + y_lfeb
        z_moe = self.moe_ln(x)
        z_moe = self.moe_conv1(z_moe)
        z_moe = self.moe_sg(z_moe)                  
        z_moe, moe_loss = self.moe_module(z_moe)     
        z_moe = self.moe_conv2(z_moe)
        fused_feat = torch.cat([y_lfeb, z_moe], dim=1)
        out = y_lfeb + self.fusion_conv(fused_feat)
        return out, moe_loss    


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=1
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class ProcessBlock(nn.Module):
    def __init__(
        self,
        dims,
        d_state=16,
        n_l_block=1,
        n_h_block=1,
        LayerNorm_type='WithBias'
    ):
        super(ProcessBlock,self).__init__()
        self.dim = dims
        self.dwt = DWT(fuseh=False)
        self.idwt = IDWT()
        self.lnum = n_l_block
        self.hnum = n_h_block
        self.hhenhance = Depth_conv(self.dim, self.dim)
        self.llenhance = nn.ModuleList()
        for layer in range(2):
            self.llenhance.append(WTConv2d(dims, dims, kernel_size=5, wt_levels=3))

        self.hhmamba = nn.ModuleList()
        self.norm2 = LayerNorm(self.dim, LayerNorm_type)

        for layer in range(self.hnum):
            self.hhmamba.append(nn.ModuleList([
                SS2D6(d_model=dims, dropout=0, d_state=d_state, scan_type='lh'),
                PreNorm(dims, FeedForward(dim=dims))
            ]))

        self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv()

        self.posenhance = nn.ModuleList()
        for layer in range(self.lnum):
            self.posenhance.append(LightBlock(self.dim))

        self.conv_fusechannel = nn.Conv2d(self.dim*2, self.dim, 1, stride=1, bias=False)

        # 只保留接收外部传入 3 通道先验的通道映射和掩码网络
        self.hf_dwconv = nn.Conv2d(3, self.dim, kernel_size=3, padding=1) # 接收外部 3 通道梯度
        self.hf_gate_conv = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        
        self.fuse_alpha = nn.Parameter(torch.ones(1) * 0.06)
        self.prior_s_proj = nn.Conv2d(3, self.dim, kernel_size=1) # 将外部 3 通道结构投影到特征通道

    def create_conv_layer(self, kernel):
        conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1, bias=False)
        # 🌟 修复点 3：强制封装为 Parameter，杜绝 DataParallel 漏检报错
        conv.weight = nn.Parameter(kernel.repeat(self.dim, self.dim, 1, 1))
        return conv

    def create_wave_conv(self):
        horizontal_kernel = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        vertical_kernel = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        diagonal_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.create_conv_layer(horizontal_kernel), self.create_conv_layer(vertical_kernel), self.create_conv_layer(diagonal_kernel)

    def forward(self, x, prior_s=None, prior_g=None):
        b, c, h, w = x.shape
        xori = x
        
        ll, hl, lh, hh_wave = self.dwt(x)

        for layer in self.llenhance:
            ll = layer(ll)

        if prior_g is not None:
            # 尺寸对齐到小波分量
            prior_g_dwt = F.interpolate(prior_g, size=(hl.shape[2], hl.shape[3]), mode='bilinear', align_corners=False)
            F_dw = self.hf_dwconv(prior_g_dwt)
            Ms = torch.sigmoid(self.hf_gate_conv(F_dw))
            hf_blend = 0.38 + 0.62 * Ms
            hl, lh, hh_wave = hl * hf_blend, lh * hf_blend, hh_wave * hf_blend

        hh_cat = self.hhenhance(torch.cat((hl, lh, hh_wave), dim=0))
        e_hl, e_lh, e_hh = hh_cat[:b, ...], hh_cat[b:2 * b, ...], hh_cat[2 * b:, ...]

        e_hl = torch.cat((e_hl, self.horizontal_conv(ll)), dim=1)
        e_lh = torch.cat((e_lh, self.vertical_conv(ll)), dim=1)
        e_hh = torch.cat((e_hh, self.diagonal_conv(ll)), dim=1)

        e_high = self.conv_fusechannel(torch.cat((e_hl, e_lh, e_hh), dim=0))
        e_high = self.norm2(e_high)

        for (ss2d, ff) in self.hhmamba:
            y = e_high.permute(0, 2, 3, 1)
            e_high = ss2d(y) + e_high.permute(0, 2, 3, 1)
            e_high = ff(e_high) + e_high
            e_high = e_high.permute(0, 3, 1, 2)
            
        x_out = self.idwt(torch.cat((ll, e_high), dim=0)) + xori
        
        if prior_s is not None:
            prior_s_feat = self.prior_s_proj(prior_s)
            x_out = x_out + self.fuse_alpha * prior_s_feat

        moe_balance, moe_sparsity = 0.0, 0.0
        for layer in self.posenhance:
            x_out, m_loss = layer(x_out)
            moe_balance += m_loss['moe_balance']
            moe_sparsity += m_loss['moe_sparsity']

        return x_out, {'moe_balance': moe_balance, 'moe_sparsity': moe_sparsity}

class DCENet(nn.Module):
    def __init__(
        self,
        nc,
        n_l_blocks,
        n_h_blocks,
        **kwargs 
    ):
        super(DCENet,self).__init__()
        
        self.smgm_extractor = MCSM(input_channels=3, feature_channels=16)
        self.grad_loss = GradientLoss(lambda1=1.0, lambda2=0.07, threshold=0.1)
        self.gt_grad_extractor = GradientPrior() # 🌟 单独为高频真值实例化安全的提取器

        self.conv0 = nn.Conv2d(3,nc,1,1,0)
        self.conv1 = ProcessBlock(nc, d_state=16, n_l_block=n_l_blocks[0], n_h_block=n_h_blocks[0])
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = ProcessBlock(nc * 2, d_state=16, n_l_block=n_l_blocks[1], n_h_block=n_h_blocks[1])
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = ProcessBlock(nc * 3, d_state=16, n_l_block=n_l_blocks[2], n_h_block=n_h_blocks[2])
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = ProcessBlock(nc * 2, d_state=16, n_l_block=n_l_blocks[3], n_h_block=n_h_blocks[3])
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = ProcessBlock(nc, d_state=16, n_l_block=n_l_blocks[4], n_h_block=n_h_blocks[4])
        self.convout = nn.Conv2d(nc,3,1,1,0)

    def forward(self, x, gt=None):
        x_ori = x

  
        gamma, Si_prior, Gi_prior = self.smgm_extractor(x)

        smgm_loss_total = torch.tensor(0.0, device=x.device)
        if gt is not None and next(self.smgm_extractor.parameters()).requires_grad:
            gt_grad = self.gt_grad_extractor(gt) # 🌟 统一采用安全的提边模块
            smgm_loss_total = self.grad_loss(Gi_prior, gt_grad.detach())

        
        Si_0, Gi_0 = Si_prior, Gi_prior
        Si_1 = F.interpolate(Si_0, scale_factor=0.5, mode='bilinear', align_corners=False)
        Gi_1 = F.interpolate(Gi_0, scale_factor=0.5, mode='bilinear', align_corners=False)
        Si_2 = F.interpolate(Si_1, scale_factor=0.5, mode='bilinear', align_corners=False)
        Gi_2 = F.interpolate(Gi_1, scale_factor=0.5, mode='bilinear', align_corners=False)

       
        x = self.conv0(x)
        
        x01, loss1 = self.conv1(x, prior_s=Si_0, prior_g=Gi_0)
        x1 = self.downsample1(x01)
        
        x12, loss2 = self.conv2(x1, prior_s=Si_1, prior_g=Gi_1)
        x2 = self.downsample2(x12)
        
        x3, loss3 = self.conv3(x2, prior_s=Si_2, prior_g=Gi_2)
        
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4_up, loss4 = self.conv4(x34, prior_s=Si_1, prior_g=Gi_1) 
        
        x5_in = self.up2(torch.cat([F.interpolate(x4_up,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5, loss5 = self.conv5(x5_in, prior_s=Si_0, prior_g=Gi_0)
        
        xout = self.convout(x5)
        xout = x_ori + xout

        
        total_balance = loss1['moe_balance'] + loss2['moe_balance'] + loss3['moe_balance'] + loss4['moe_balance'] + loss5['moe_balance']
        total_sparsity = loss1['moe_sparsity'] + loss2['moe_sparsity'] + loss3['moe_sparsity'] + loss4['moe_sparsity'] + loss5['moe_sparsity']
        moe_losses = {'moe_balance': total_balance, 'moe_sparsity': total_sparsity}

        
        smgm_dict = {'smgm_loss': smgm_loss_total}

        return xout, moe_losses, smgm_dict