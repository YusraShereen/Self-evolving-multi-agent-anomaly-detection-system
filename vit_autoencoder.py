
import torch
import torch.nn as nn
import timm
from typing import Tuple


class ViTAutoencoder(nn.Module):
    """Vision Transformer Autoencoder for reconstruction-based anomaly detection"""
    
    def __init__(self, model_name: str = 'vit_tiny_patch16_224', pretrained: bool = True):
        super().__init__()
        
        # Encoder: Pre-trained ViT
        self.encoder = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Keep patch tokens
        )
        
        self.patch_size = 16
        
        # Determine feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.encoder(dummy)
            self.feat_dim = feat.shape[-1]
            self.num_patches = (224 // self.patch_size) ** 2
        
        # Decoder: Reconstruct patches
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feat_dim * 2, self.patch_size * self.patch_size * 3)
        )
        
        print(f"  ViT Autoencoder initialized:")
        print(f"    Encoder: {model_name}")
        print(f"    Feature dim: {self.feat_dim}")
        print(f"    Patches: {self.num_patches}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            reconstructed_patches: [B, N, 3, P, P]
            encoded_features: [B, N+1, D]
        """
        # Encode
        encoded = self.encoder(x)  # [B, N+1, D] where N=num_patches
        
        # Skip CLS token
        patch_features = encoded[:, 1:, :]  # [B, N, D]
        
        # Decode each patch
        reconstructed = self.decoder(patch_features)  # [B, N, P*P*3]
        
        # Reshape to patch format
        B, N = reconstructed.shape[:2]
        reconstructed = reconstructed.view(B, N, 3, self.patch_size, self.patch_size)
        
        return reconstructed, encoded

    def compute_reconstruction_error(self, original: torch.Tensor, 
                                    reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute per-patch MSE reconstruction error
        
        Args:
            original: Original images [B, 3, 224, 224]
            reconstructed: Reconstructed patches [B, N, 3, P, P]
            
        Returns:
            patch_errors: [B, N] tensor of patch-wise errors
        """
        # Extract patches from original image
        patches = original.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(
            original.shape[0], 3, -1, self.patch_size, self.patch_size
        ).permute(0, 2, 1, 3, 4)  # [B, N, 3, P, P]
        
        # Compute MSE per patch
        mse_per_patch = ((patches - reconstructed) ** 2).mean(dim=(2, 3, 4))  # [B, N]
        
        return mse_per_patch

    def get_model_info(self) -> dict:
        """Get model architecture information"""
        return {
            "encoder": self.encoder.__class__.__name__,
            "feature_dim": self.feat_dim,
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


