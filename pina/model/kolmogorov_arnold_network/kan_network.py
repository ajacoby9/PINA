"""Kolmogorov Arnold Network implementation"""
import torch
import torch.nn as nn
from typing import List

try:
    from .kan_layer import KAN_layer
except ImportError:
    from kan_layer import KAN_layer

class KAN_Network(torch.nn.Module):
    """
    Kolmogorov Arnold Network - A neural network using KAN layers instead of traditional MLP layers.
    Each layer uses learnable univariate functions (B-splines + base functions) on edges.
    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        k: int = 3,
        num: int = 5,
        grid_eps: float = 0.02,
        grid_range: List[float] = [-1, 1],
        grid_extension: bool = True,
        noise_scale: float = 0.5,
        base_function = torch.nn.SiLU(),
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0
    ):
        """
        Initialize the KAN network.
        
        Args:
            layer_sizes: List of integers defining the size of each layer [input_dim, hidden1, hidden2, ..., output_dim]
            k: Order of the B-spline
            num: Number of grid points for B-splines
            grid_eps: Epsilon for grid spacing
            grid_range: Range for the grid [min, max]
            grid_extension: Whether to extend the grid
            noise_scale: Scale for initialization noise
            base_function: Base activation function (e.g., SiLU)
            scale_base_mu: Mean for base function scaling
            scale_base_sigma: Std for base function scaling
            scale_sp: Scale for spline functions
        """
        super().__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output dimensions")
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Create KAN layers
        self.kan_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer = KAN_layer(
                k=k,
                input_dimensions=layer_sizes[i],
                output_dimensions=layer_sizes[i+1],
                inner_nodes=0,  # Not used in current implementation
                num=num,
                grid_eps=grid_eps,
                grid_range=grid_range,
                grid_extension=grid_extension,
                noise_scale=noise_scale,
                base_function=base_function,
                scale_base_mu=scale_base_mu,
                scale_base_sigma=scale_base_sigma,
                scale_sp=scale_sp
            )
            self.kan_layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dimensions)
            
        Returns:
            Output tensor of shape (batch_size, output_dimensions)
        """
        current = x
        
        for layer in self.kan_layers:
            current = layer(current)
            
        return current
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def update_grid_resolution(self, new_num: int):
        """
        Update the grid resolution for all layers.
        This can be used for adaptive training where grid resolution increases over time.
        
        Args:
            new_num: New number of grid points
        """
        for layer in self.kan_layers:
            # This would require implementing grid refinement logic
            # For now, this is a placeholder for future enhancement
            layer.num = new_num
            # TODO: Implement actual grid refinement

        pass
            
    def enable_sparsification(self, threshold: float = 1e-4):
        """
        Enable sparsification by setting small weights to zero.
        
        Args:
            threshold: Threshold below which weights are set to zero
        """
        with torch.no_grad():
            for layer in self.kan_layers:
                # Sparsify scale parameters
                layer.scale_base.data[torch.abs(layer.scale_base.data) < threshold] = 0
                layer.scale_spline.data[torch.abs(layer.scale_spline.data) < threshold] = 0
                
                # Update mask
                layer.mask = ((torch.abs(layer.scale_base) >= threshold) | 
                             (torch.abs(layer.scale_spline) >= threshold)).float()

    def get_activation_statistics(self, x: torch.Tensor):
        """
        Get statistics about activations for analysis purposes.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with activation statistics
        """
        stats = {}
        current = x
        
        for i, layer in enumerate(self.kan_layers):
            current = layer(current)
            stats[f'layer_{i}'] = {
                'mean': current.mean().item(),
                'std': current.std().item(),
                'min': current.min().item(),
                'max': current.max().item()
            }
            
        return stats
