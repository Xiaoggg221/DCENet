import models.archs.DCENet as DCENet

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    nf = opt_net['nf']
    n_l_blocks = opt_net['n_l_blocks']
    n_h_blocks = opt_net['n_h_blocks']

    if which_model == 'DCENet':
        train_opt = opt.get('train', {})
        smgm_loss_agg = train_opt.get('smgm_loss_agg', 'mean')
        smgm_stage_weights = train_opt.get('smgm_stage_weights', None)
        save_prior_debug_train = train_opt.get('save_prior_debug', True)
        netG = DCENet.DCENet(
            nc=nf,
            n_l_blocks=n_l_blocks,
            n_h_blocks=n_h_blocks,
            smgm_loss_agg=smgm_loss_agg,
            smgm_stage_weights=smgm_stage_weights,
            save_prior_debug_train=save_prior_debug_train,
        )
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

