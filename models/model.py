from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class Autoencoder(nn.Module):
    def __init__(self, encoder: Module, decoder: Module, latent_layer: Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_layer = latent_layer

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.decoder(self.latent_layer(self.encoder(x, **kwargs)))


class ConvolutionalBasicEncoder(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super(ConvolutionalBasicEncoder, self).__init__()
        layers = []
        init_dim = 3
        for d in dims:
            layers.append(
                nn.Conv2d(
                    in_channels=init_dim, out_channels=d, kernel_size=3, padding=1
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.MaxPool2d(2, 2))
            init_dim = d
        self.graph = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        return self.graph(x)


class ConvlutionalBasicDecoder(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super(ConvlutionalBasicDecoder, self).__init__()
        layers = []
        init_dim = dims[0]
        dims.append(3)
        for d in dims[1:]:
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.GELU())
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=init_dim, out_channels=d, kernel_size=3, padding=1
                )
            )            
            init_dim=d
        self.graph = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        return self.graph(x)


def create_fullyConvolutionalAutoencoder(dims:List[int]) -> Module:
    encoder = ConvolutionalBasicEncoder(dims=dims)
    decoder = ConvlutionalBasicDecoder(dims=dims[::-1])
    latent_layer = nn.Identity()
    return  Autoencoder(encoder=encoder,decoder=decoder,latent_layer=latent_layer)

dims = [32,32, 64, 64, 128, 128, 256, 512]