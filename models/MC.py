from utils import ConvSTFT, ConviSTFT
from models.baseBlocks import *


class EncoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(EncoderStage, self).__init__()
        self.stage_num = layers

        # input layer
        self.inconv = CINCONV(in_ch, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CCONV(out_ch, mid_ch) if i == 0 else CCONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = complexDilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [CSPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else CSPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])
        # attention module
        self.att = CCTFA(out_ch)

        # down-sampling block
        self.downsampling = complex_down_sampling(out_ch)

    def forward(self, xr_in, xi_in):
        xr_in, xi_in = self.inconv(xr_in, xi_in)

        out_r, out_i = xr_in, xi_in
        encoder_outs_r, encoder_outs_i = [], []
        for idx, layers in enumerate(self.encoders):
            out_r, out_i = layers(out_r, out_i)
            encoder_outs_r.append(out_r), encoder_outs_i.append(out_i)

        out_r, out_i = self.bottleneck(out_r, out_i)

        for idx, layers in enumerate(self.decoders):
            out_r, out_i = layers(torch.cat([out_r, encoder_outs_r[-idx - 1]], dim=1),
                                  torch.cat([out_i, encoder_outs_i[-idx - 1]], dim=1))
        out_r, out_i = self.att(out_r, out_i)
        out_r = out_r + xr_in
        out_i = out_i + xi_in

        out_r, out_i = self.downsampling(out_r, out_i)
        return out_r, out_i


class DecoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(DecoderStage, self).__init__()
        self.stage_num = layers

        # up-sampling block
        self.upsampling = CSPCONV(in_ch * 2, in_ch * 2)

        # input layer
        self.inconv = CINCONV(in_ch * 2, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CCONV(out_ch, mid_ch) if i == 0 else CCONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = complexDilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [CSPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else CSPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])

        self.att = CCTFA(out_ch)

    def forward(self, xr_in, xi_in):
        xr_in, xi_in = self.upsampling(xr_in, xi_in)

        xr_in, xi_in = self.inconv(xr_in, xi_in)

        out_r, out_i = xr_in, xi_in
        encoder_outs_r, encoder_outs_i = [], []
        for idx, layers in enumerate(self.encoders):
            out_r, out_i = layers(out_r, out_i)
            encoder_outs_r.append(out_r), encoder_outs_i.append(out_i)

        out_r, out_i = self.bottleneck(out_r, out_i)

        for idx, layers in enumerate(self.decoders):
            out_r, out_i = layers(torch.cat([out_r, encoder_outs_r[-idx - 1]], dim=1),
                                  torch.cat([out_i, encoder_outs_i[-idx - 1]], dim=1))

        out_r, out_i = self.att(out_r, out_i)

        return out_r + xr_in, out_i + xi_in


class MC(nn.Module):
    def __init__(self, in_ch=2, mid_ch=64, out_ch=128,
                 WIN_LEN=512, HOP_LEN=256, FFT_LEN=512):
        super(MC, self).__init__()
        self.fft_half = FFT_LEN // 2 + 1

        # Input layer
        self.input_layer = CINCONV(in_ch, out_ch)

        # Encoder stages
        self.en1 = EncoderStage(out_ch, mid_ch, out_ch, layers=6)
        self.en2 = EncoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.en3 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en4 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en5 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en6 = EncoderStage(out_ch, mid_ch, out_ch, layers=3)

        # Bottleneck block
        self.bottleneck = complexDilatedDenseBlock(out_ch, out_ch, 6)

        # Decoder stages
        self.de1 = DecoderStage(out_ch, mid_ch, out_ch, layers=3)
        self.de2 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de3 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de4 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de5 = DecoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.de6 = DecoderStage(out_ch, mid_ch, out_ch, layers=6)

        # output layer
        self.output_layer = COUTCONV(out_ch, in_ch)

        # for feature extract
        self.cstft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')
        self.cistft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')

    def forward(self, real, imag):
        real = real.unsqueeze(1)[:, :, 1:]
        imag = imag.unsqueeze(1)[:, :, 1:]

        # input layer
        hx_r, hx_i = self.input_layer(real, imag)

        # encoder stages
        hx1r, hx1i = self.en1(hx_r, hx_i)
        hx2r, hx2i = self.en2(hx1r, hx1i)
        hx3r, hx3i = self.en3(hx2r, hx2i)
        hx4r, hx4i = self.en4(hx3r, hx3i)
        hx5r, hx5i = self.en5(hx4r, hx4i)
        hx6r, hx6i = self.en6(hx5r, hx5i)

        # dilated dense block
        out_r, out_i = self.bottleneck(hx6r, hx6i)

        # decoder stages - masking
        out_r, out_i = self.de1(torch.cat([out_r, hx6r], dim=1), torch.cat([out_i, hx6i], dim=1))
        out_r, out_i = self.de2(torch.cat([out_r, hx5r], dim=1), torch.cat([out_i, hx5i], dim=1))
        out_r, out_i = self.de3(torch.cat([out_r, hx4r], dim=1), torch.cat([out_i, hx4i], dim=1))
        out_r, out_i = self.de4(torch.cat([out_r, hx3r], dim=1), torch.cat([out_i, hx3i], dim=1))
        out_r, out_i = self.de5(torch.cat([out_r, hx2r], dim=1), torch.cat([out_i, hx2i], dim=1))
        out_r, out_i = self.de6(torch.cat([out_r, hx1r], dim=1), torch.cat([out_i, hx1i], dim=1))

        # output layer
        real_out, imag_out = self.output_layer(out_r, out_i)

        mask_mags = torch.sqrt(real_out ** 2 + imag_out ** 2)
        phase_real = real_out / (mask_mags + 1e-8)
        phase_imag = imag_out / (mask_mags + 1e-8)
        mask_phase = torch.atan2(phase_imag, phase_real)

        mask_mags = torch.tanh(mask_mags)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        mag_out = mag * mask_mags
        phase_out = phase + mask_phase

        real_out = mag_out * torch.cos(phase_out)
        imag_out = mag_out * torch.sin(phase_out)

        mag_out = functional.pad(mag_out, [0, 0, 1, 0])
        real_out = functional.pad(real_out, [0, 0, 1, 0])
        imag_out = functional.pad(imag_out, [0, 0, 1, 0])
        return real_out.squeeze(1), imag_out.squeeze(1), mag_out.squeeze(1)