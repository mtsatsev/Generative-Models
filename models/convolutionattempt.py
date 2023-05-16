class ConvolutionalBasicAutoencoder(Autoencoder):
    def __init__(self, dims: List[int],latent_dim: int) -> None:
        layers = []
        init_channels = 3 
        for dim in dims:
            layers.append(
                nn.Conv2d(in_channels=init_channels,out_channels=dim,kernel_size=3,padding=1)
            )
            layers.append(nn.MaxPool2d(kernel_size=1))
            init_channels=dim
        encoder = nn.Sequential(*layers)
        latent_layer = nn.Conv2d(in_channels=dims[-1],out_channels=latent_dim,kernel_size=1)
        super().__init__(encoder=encoder,decoder=encoder,latent_layer=latent_layer)

    def forward(self, x: Tensor) -> Tensor:
        z = self.latent_linear(
            self.encoder(
                x
            )
        )
        return z
    
