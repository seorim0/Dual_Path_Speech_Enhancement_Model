# get architecture
def get_arch(opt):
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'SM':
        from models.SM import SM
        model = SM(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                   WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    elif arch == 'MM':
        from models.MM import MM
        model = MM(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                   WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    elif arch == 'SC':
        from models.SC import SC
        model = SC(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                   WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    elif arch == 'MC':
        from models.MC import MC
        model = MC(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                   WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    elif arch == 'MSC':
        from models.MSC import MSC
        model = MSC(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                    WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    elif arch == 'DPCNU':
        from models.DPCNU import DPCNU
        model = DPCNU(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                      WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    else:
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    loss_type = opt.loss_type

    print('You choose ' + loss_type + '...')

    if loss_type == 'mag':
        from .trainer import mag_loss_train
        from .trainer import mag_loss_valid
        trainer = mag_loss_train
        validator = mag_loss_valid
    elif loss_type == 'mag+real+imag':  # multiple(joint) loss function
        from .trainer import mag_real_imag_loss_train
        from .trainer import mag_real_imag_loss_valid
        trainer = mag_real_imag_loss_train
        validator = mag_real_imag_loss_valid
    else:
        raise Exception("Loss type error!")

    return trainer, validator


def get_loss(opt):
    from torch.nn import L1Loss
    from torch.nn.functional import mse_loss
    loss_oper = opt.loss_oper

    print('You choose loss operation with ' + loss_oper + '...')
    if loss_oper == 'l1':
        loss_calculator = L1Loss()
    elif loss_oper == 'l2':
        loss_calculator = mse_loss
    else:
        raise Exception("Arch error!")

    return loss_calculator
