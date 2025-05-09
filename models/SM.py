from utils import ConvSTFT, ConviSTFT
from models.baseBlocks import *


class EncoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(EncoderStage, self).__init__()
        self.stage_num = layers

        # input layer
        self.inconv = INCONV(in_ch, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CONV(out_ch, mid_ch) if i == 0 else CONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = dilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [SPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else SPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])
        # attention module
        self.att = CTFA(out_ch)

        # down-sampling block
        self.downsampling = down_sampling(out_ch)

    def forward(self, x_in):
        x_in = self.inconv(x_in)

        out = x_in
        encoder_outs = []
        for idx, layers in enumerate(self.encoders):
            out = layers(out)
            encoder_outs.append(out)

        out = self.bottleneck(out)

        for idx, layers in enumerate(self.decoders):
            out = layers(torch.cat([out, encoder_outs[-idx - 1]], dim=1))

        out = self.att(out) + x_in

        out = self.downsampling(out)
        return out


class DecoderStage(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layers=4, bottleneck_layers=6):
        super(DecoderStage, self).__init__()
        self.stage_num = layers

        # up-sampling block
        self.upsampling = SPCONV(in_ch * 2, in_ch * 2)

        # input layer
        self.inconv = INCONV(in_ch * 2, out_ch)
        # inner encoders
        self.encoders = nn.ModuleList(
            [CONV(out_ch, mid_ch) if i == 0 else CONV(mid_ch, mid_ch) for i in range(self.stage_num)])
        # inner bottleneck
        self.bottleneck = dilatedDenseBlock(mid_ch, mid_ch, bottleneck_layers, inner=True)
        # inner decoders
        self.decoders = nn.ModuleList(
            [SPCONV(mid_ch * 2, out_ch) if i == self.stage_num - 1 else SPCONV(mid_ch * 2, mid_ch) for i in
             range(self.stage_num)])

        self.att = CTFA(out_ch)

    def forward(self, x_in):
        x_in = self.upsampling(x_in)

        x_in = self.inconv(x_in)

        out = x_in
        encoder_outs = []
        for idx, layers in enumerate(self.encoders):
            out = layers(out)
            encoder_outs.append(out)

        out = self.bottleneck(out)

        for idx, layers in enumerate(self.decoders):
            out = layers(torch.cat([out, encoder_outs[-idx - 1]], dim=1))

        out = self.att(out)

        return out + x_in


class SM(nn.Module):

    def __init__(self, in_ch=1, mid_ch=32, out_ch=64,
                 WIN_LEN=400, HOP_LEN=100, FFT_LEN=512):
        super(SM, self).__init__()
        self.fft_half = FFT_LEN // 2 + 1

        # Input layer
        self.input_layer = INCONV(in_ch, out_ch)

        # Encoder stages
        self.en1 = EncoderStage(out_ch, mid_ch, out_ch, layers=6)
        self.en2 = EncoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.en3 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en4 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en5 = EncoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.en6 = EncoderStage(out_ch, mid_ch, out_ch, layers=3)

        # Bottleneck block
        self.bottleneck = dilatedDenseBlock(out_ch, out_ch, 6)

        # Decoder stages
        self.de1 = DecoderStage(out_ch, mid_ch, out_ch, layers=3)
        self.de2 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de3 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de4 = DecoderStage(out_ch, mid_ch, out_ch, layers=4)
        self.de5 = DecoderStage(out_ch, mid_ch, out_ch, layers=5)
        self.de6 = DecoderStage(out_ch, mid_ch, out_ch, layers=6)

        # output layer
        self.output_layer = nn.Conv2d(out_ch, in_ch, kernel_size=1)

        # for feature extract
        self.cstft = ConvSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')
        self.cistft = ConviSTFT(WIN_LEN, HOP_LEN, FFT_LEN, feature_type='complex')

    def forward(self, real, imag):
        # STFT
        mags = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)
        hx = mags.unsqueeze(1)  # [B, 1, F, T]
        hx = hx[:, :, 1:]

        # input layer
        hx = self.input_layer(hx)

        # encoder stages
        hx1 = self.en1(hx)
        hx2 = self.en2(hx1)
        hx3 = self.en3(hx2)
        hx4 = self.en4(hx3)
        hx5 = self.en5(hx4)
        hx6 = self.en6(hx5)

        # dilated dense block
        out = self.bottleneck(hx6)

        # decoder stages - direct
        out = self.de1(torch.cat([out, hx6], dim=1))
        out = self.de2(torch.cat([out, hx5], dim=1))
        out = self.de3(torch.cat([out, hx4], dim=1))
        out = self.de4(torch.cat([out, hx3], dim=1))
        out = self.de5(torch.cat([out, hx2], dim=1))
        out = self.de6(torch.cat([out, hx1], dim=1))

        # output layer
        out = self.output_layer(out)

        out = functional.pad(out, [0, 0, 1, 0]).squeeze(1)

        real_out = out * torch.cos(phase)
        imag_out = out * torch.sin(phase)

        return real_out, imag_out, out

