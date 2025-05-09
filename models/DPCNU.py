from torch import nn
from MSC import MSC
from MC import MC
from utils import power_uncompress


class DPCNU(nn.Module):
    def __init__(self, in_ch=2, mid_ch=64, out_ch=128,
                 WIN_LEN=512, HOP_LEN=256, FFT_LEN=512):
        super(DPCNU, self).__init__()

        self.main_path = MSC(in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch,
                             WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN, FFT_LEN=FFT_LEN)
        self.auxiliary_path = MC(in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch,
                                 WIN_LEN=WIN_LEN, HOP_LEN=HOP_LEN, FFT_LEN=FFT_LEN)

    def forward(self, real, imag):
        # main path
        main_out_real, main_out_imag, _ = self.main_path(real, imag)
        main_out_specs = power_uncompress(main_out_real, main_out_imag)

        main_outputs = self.main_path.cistft(main_out_specs)
        main_outputs = main_outputs.squeeze(1)

        # auxiliary path
        auxiliary_out_real, auxiliary_out_imag, _ = self.auxiliary_path(real, imag)
        auxiliary_out_specs = power_uncompress(auxiliary_out_real, auxiliary_out_imag)

        auxiliary_outputs = self.auxiliary_path.cistft(auxiliary_out_specs)
        auxiliary_outputs = auxiliary_outputs.squeeze(1)

        outputs = (main_outputs + auxiliary_outputs) / 2

        return outputs