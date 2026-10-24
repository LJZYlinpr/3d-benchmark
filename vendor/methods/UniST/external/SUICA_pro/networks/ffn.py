import torch
import torch.nn as nn
import numpy as np

class GaussianEncoding(nn.Module):
    """
    Given an input of size [batches, num_input_channels],
     returns a tensor of size [batches, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._scale = scale
        
        # Initialize B matrix (fixed encoding matrix)
        B = torch.randn((num_input_channels, mapping_size)) * scale
        # Fixed encoding matrix (use register_buffer for device management)
        self.register_buffer('B', B)

    def forward(self, x):
        x = x @ self.B

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class MultiScaleGaussianEncoding(nn.Module):
    """
    Multi-scale Fourier feature encoding - uses multiple frequency ranges
    to improve representation capability for different regions.
    Particularly suitable for handling complex regions like cracks.
    """
    def __init__(self, num_input_channels, mapping_size=128, scales=[1, 10, 100]):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.scales = scales
        
        self.encodings = nn.ModuleList([
            GaussianEncoding(num_input_channels, mapping_size, scale=s)
            for s in scales
        ])

    def forward(self, x):
        # Encode at each scale and concatenate
        encoded = [enc(x) for enc in self.encodings]
        return torch.cat(encoded, dim=1)


class EnhancedGaussianEncoding(nn.Module):
    """
    Enhanced Fourier feature encoding
    - Multi-scale frequency encoding
    - Supports anisotropic encoding (uses different frequencies for z-direction)
    """
    def __init__(self, num_input_channels, mapping_size=128, scales=[1, 10, 100], 
                 anisotropic_3d=False, z_scales=None):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.mapping_size = mapping_size
        self.scales = scales
        self.anisotropic_3d = anisotropic_3d and num_input_channels == 3
        
        # If 3D and anisotropic encoding is needed, use different frequencies for z-direction
        if self.anisotropic_3d:
            if z_scales is None:
                # Default: use lower frequencies for z-direction (because z-direction is sparse)
                z_scales = [s * 0.1 for s in scales]  # Reduce z-direction frequency by 10x
            self.z_scales = z_scales
            
            # xy-direction encoding (first 2 dimensions)
            self.xy_encodings = nn.ModuleList([
                GaussianEncoding(2, mapping_size, scale=s)
                for s in scales
            ])
            # z-direction encoding (3rd dimension)
            self.z_encodings = nn.ModuleList([
                GaussianEncoding(1, mapping_size, scale=s)
                for s in z_scales
            ])
        else:
            # Standard multi-scale encoding
            self.encodings = nn.ModuleList([
                GaussianEncoding(num_input_channels, mapping_size, scale=s)
                for s in scales
            ])

    def forward(self, x):
        # Anisotropic encoding (3D, sparse z-direction)
        if self.anisotropic_3d:
            # Separate xy and z coordinates
            xy = x[:, :2]  # [batch, 2]
            z = x[:, 2:3]  # [batch, 1]
            
            # xy-direction encoding
            xy_encoded = [enc(xy) for enc in self.xy_encodings]
            # z-direction encoding (using different frequencies)
            z_encoded = [enc(z) for enc in self.z_encodings]
            
            # Concatenate all encodings
            encoded = xy_encoded + z_encoded
            result = torch.cat(encoded, dim=1)
        else:
            # Standard multi-scale encoding
            encoded = [enc(x) for enc in self.encodings]
            result = torch.cat(encoded, dim=1)
        
        return result


class FourierFeatureNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, final_activation,
                 encoding_type='basic', mapping_size=256, encoding_scales=[1, 10, 100],
                 anisotropic_3d=False, z_scales=None, network_configs=None):
        super().__init__()
        
        # If network_configs is provided, prioritize parameters from it
        if network_configs is not None:
            anisotropic_3d = getattr(network_configs, 'anisotropic_3d', anisotropic_3d)
            z_scales = getattr(network_configs, 'z_scales', z_scales)
        
        # Select encoding type
        if encoding_type == 'basic':
            self.transform = GaussianEncoding(dim_in, mapping_size, scale=10)
            encoding_dim = mapping_size * 2
        elif encoding_type == 'multiscale':
            self.transform = MultiScaleGaussianEncoding(dim_in, mapping_size, scales=encoding_scales)
            encoding_dim = mapping_size * 2 * len(encoding_scales)
        elif encoding_type == 'enhanced':
            self.transform = EnhancedGaussianEncoding(
                dim_in, mapping_size, scales=encoding_scales,
                anisotropic_3d=anisotropic_3d,
                z_scales=z_scales
            )
            
            # Calculate encoding dimension
            if anisotropic_3d and dim_in == 3:
                # xy-direction + z-direction encoding (each has mapping_size*2 dimensions)
                encoding_dim = mapping_size * 2 * len(encoding_scales) * 2  # One set for xy and one for z
            else:
                encoding_dim = mapping_size * 2 * len(encoding_scales)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
        
        self.encoding_dim = encoding_dim
        final_activation = getattr(nn, final_activation)()

        self.layers = nn.Sequential(
                        nn.Linear(encoding_dim, dim_hidden),
                        nn.ReLU(),
                        *[nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers)],
                        nn.Linear(dim_hidden, dim_out),
                        final_activation
                    )
        
    
    def forward(self, x):
        ff = self.transform(x)
        y = self.layers(ff)
        return y


if __name__ == "__main__":
    # Test basic version
    transform_basic = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2, 
                                       final_activation="Identity", encoding_type='basic')
    x = torch.ones([100, 2])
    y = transform_basic(x)
    print(f"Basic encoding output shape: {y.shape}")
    
    # Test multi-scale version
    transform_multiscale = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2,
                                            final_activation="Identity", encoding_type='multiscale',
                                            mapping_size=128, encoding_scales=[1, 10, 100])
    y = transform_multiscale(x)
    print(f"Multiscale encoding output shape: {y.shape}")
    
    # Test enhanced version (recommended for crack regions)
    transform_enhanced = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2,
                                          final_activation="Identity", encoding_type='enhanced',
                                          mapping_size=128, encoding_scales=[1, 10, 100])
    y = transform_enhanced(x)
    print(f"Enhanced encoding output shape: {y.shape}")