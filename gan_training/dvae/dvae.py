import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean=0, std=0.02)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps  


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, c_dim=10, nc=3, infodistil_mode=False):
        super(Encoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.infodistil_mode = infodistil_mode
        self.conv_layer = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True)
        )
        self.fc_layer = nn.Linear(256 * 4 * 4, c_dim*2)

    def forward(self, x):
        if self.infodistil_mode:
            x = x.add(1).div(2)
            if (x.size(2) > 64) or (x.size(3) > 64):
                x = F.adaptive_avg_pool2d(x, (64, 64))

        batch_size = x.size(0)
        h = self.conv_layer(x)
        h = h.view(batch_size, -1)
        h = self.fc_layer(h)
        return h


class Decoder(nn.Module):
    def __init__(self, c_dim=10, nc=3):
        super(Decoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Linear(c_dim, 256 * 7 * 7),  # B, 256
            View((-1, 256, 7, 7)),               # B, 256,  7,  7
            nn.ReLU(True),
            upsample,
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(16, nc, 1)
        )

    def forward(self, c):
        x = self.layer(c)
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim, size=64, nc=3):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.layer = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, 1),             # B, 1
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=10, size=64, nc=3, **kwarngs):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),  # B, 256
            View((-1, 256, 7, 7)),  # B, 256,  7,  7
            nn.ReLU(True),
            upsample,
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            upsample,
            nn.Conv2d(16, nc, 1),
            nn.Tanh()
        )

    def forward(self, c):
        x = self.layer(c)
        return x


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, c_dim=10, nc=3, infodistil_mode=False):
        super(BetaVAE_H, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.encoder = Encoder(c_dim, nc, infodistil_mode)
        self.decoder = Decoder(c_dim, nc)
        self.apply(normal_init)

    def forward(self, x, c, encode_only, decode_only):
        if encode_only:
            c, mu, logvar = self._encode(x)
            return c, mu, logvar
        elif decode_only:
            x_recon = self._decode(c)
            return x_recon
        else:
            c, mu, logvar = self._encode(x)
            x_recon = self._decode(c)
            return x_recon, c, mu, logvar

    def __call__(self, x=None, c=None, encode_only=False, decode_only=False):
        return self.forward(x, c, encode_only, decode_only)

    def _encode(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.c_dim]
        logvar = distributions[:, self.c_dim:]
        c = reparametrize(mu, logvar)
        return c, mu, logvar

    def _decode(self, c):
        return self.decoder(c)


if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c_dim = 20
    nc = 3
    vae = BetaVAE_H(c_dim, nc).to(device)

    input = torch.rand(12, 3, 112, 112).to(device)
    x_recon, c, mu, logvar = vae(input)
    print(x_recon.size())
    print(c.size())
    print(mu.size())
    print(logvar.size())

    generator = Generator(z_dim=c_dim, nc=nc).to(device)
    output = generator(c)
    print(output.size())