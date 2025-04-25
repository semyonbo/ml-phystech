import math
import librosa
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import soundfile as sf
from skimage.transform import resize
from scipy.signal import butter, filtfilt
from torch.nn.utils import spectral_norm
import os
import matplotlib.pyplot as plt
import imageio
from torchvision.utils import save_image




def load_files(bird_type):
    df = pd.read_csv('bird_songs_metadata.csv')
    df_species = df if bird_type=='all' else df[df["name"] == bird_type]
    path = 'wavfiles/'
    audios = []
    for fn in df_species["filename"]:
        p = os.path.join(path, fn)
        if os.path.exists(p):
            y,_ = librosa.load(p, sr=22050)
            # b,a = butter(6, [1000/11025, 8000/11025], btype='band')
            # y = filtfilt(b, a, y)
            audios.append(y)
    return np.array(audios)


def audio_to_mel_spectrogram(audios, n_mels, shape, sr=22050):
    specs = []
    for y in audios:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S = librosa.power_to_db(S, ref=np.max)
        S = resize(S, shape, anti_aliasing=True)
        specs.append(S)
    specs = np.stack(specs, 0)
    gmin, gmax = specs.min(), specs.max()
    specs = 2*(specs - gmin)/(gmax - gmin) - 1
    return specs, gmin, gmax


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key   = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels,    1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h*w).permute(0,2,1)   # b×(h·w)×(c/8)
        k = self.key(x).view(b, -1, h*w)                    # b×(c/8)×(h·w)
        attn = F.softmax(torch.bmm(q, k), dim=-1)           # b×(h·w)×(h·w)
        v = self.value(x).view(b, -1, h*w)                  # b×c×(h·w)
        out = torch.bmm(v, attn.permute(0,2,1)).view(b,c,h,w)
        return self.gamma * out + x


class Generator(nn.Module):
    def __init__(self, latent_dim, shape):
        super().__init__()
        H, W = shape
        assert H == W and H >= 8 and (H & (H-1)) == 0, \
            "shape must be a tuple (N,N) where N is power of 2 and ≥ 8"
        self.init_size = 4
        self.num_ups = int(math.log2(H // self.init_size))
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size**2)

        layers = [nn.BatchNorm2d(512)]
        in_ch = 512
        for _ in range(self.num_ups):
            out_ch = in_ch // 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if out_ch == 128:
                layers.append(SelfAttention(out_ch))
            in_ch = out_ch

        layers += [nn.Conv2d(in_ch, 1, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.l1(z).view(z.size(0), 512, self.init_size, self.init_size)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, shape):
        super().__init__()
        H, W = shape
        assert H == W and H >= 8 and (H & (H-1)) == 0, \
            "shape must be a tuple (N,N) where N is power of 2 and ≥ 8"
        self.num_down = int(math.log2(H // 4))

        layers = []
        in_c = 1
        out_list = [16, 32, 64, 128, 256, 512]
        for idx in range(self.num_down):
            out_c = out_list[idx]
            layers += [
                spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if out_c == 32:
                layers.append(SelfAttention(out_c))
            in_c = out_c

        layers += [
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(in_c * 4 * 4, 1),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_gan(G, D, loader, latent_dim, device, epochs, lr=1e-4):
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=7*lr/8, betas=(0.5,0.999))

    G_losses = []
    D_losses = []
    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('loss_plots', exist_ok=True)

    for ep in range(1, epochs+1):
        for real, in loader:
            real = real.to(device)
            bs   = real.size(0)

            # D-step
            z    = torch.randn(bs, latent_dim, device=device)
            fake = G(z).detach()
            loss_D = (criterion(D(real), torch.ones(bs,1,device=device)) +
                      criterion(D(fake), torch.zeros(bs,1,device=device)))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # G-step
            z      = torch.randn(bs, latent_dim, device=device)
            fake2  = G(z)
            loss_G = criterion(D(fake2), torch.ones(bs,1,device=device))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            
        with torch.no_grad():
            fixed_noise = torch.randn(1, latent_dim, device=device)
            fake = G(fixed_noise).detach().cpu()
            img_path = f'generated_images/epoch_{ep:03d}.png'
            save_image(fake, img_path, normalize=True)

        print(f"[{ep}/{epochs}] D_loss={loss_D.item():.4f}  G_loss={loss_G.item():.4f}")
    
    plt.figure(figsize=(10,5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Losses during training")
    plt.legend()
    plt.savefig('loss_plots/loss_curve.png')
    plt.close()
 
    images = []
    for epoch in range(1, epochs + 1):
        img_path = f"generated_images/epoch_{epoch:03d}.png"
        images.append(imageio.imread(img_path))

    imageio.mimsave("training_progress.gif", images, fps=8)


class BirdsongGANTrainer:
    def __init__(self, bird_type, batch_size, epochs, n_mels, shape, latent_dim=100, device=None):
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.epochs     = epochs
        
        audios = load_files(bird_type)
        specs, self.db_min, self.db_max = audio_to_mel_spectrogram(audios, n_mels, shape)
        dataset = TensorDataset(torch.tensor(specs[:,None,:,:], dtype=torch.float32))
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.G = Generator(latent_dim, shape).to(self.device)
        self.D = Discriminator(shape).to(self.device)

    def train(self):
        train_gan(self.G, self.D, self.loader,
                  latent_dim=self.latent_dim,
                  device=self.device,
                  epochs=self.epochs)

    def save_models(self, gen_path='G.pth', disc_path='D.pth'):
        torch.save(self.G.state_dict(), gen_path)
        torch.save(self.D.state_dict(), disc_path)
        
    def load_models(self, gen_path='G.pth', disc_path='D.pth'):
        self.G.load_state_dict(torch.load(gen_path, map_location=self.device))
        self.D.load_state_dict(torch.load(disc_path, map_location=self.device))

    def generate_and_save_audio(self, out='generated.wav'):
        self.G.eval()
        with torch.no_grad():
            z    = torch.randn(1, self.latent_dim, device=self.device)
            spec = self.G(z).cpu().squeeze(0).squeeze(0).numpy()
            # денормализация
            spec = (spec + 1)/2 * (self.db_max - self.db_min) + self.db_min
            audio = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(spec), sr=22050)
            sf.write(out, audio, 22050)
            print(f"[INFO] Saved: {out}")
