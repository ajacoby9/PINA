"""Create the infrastructure for a KAN layer"""
import torch
import numpy as np

try:
    from .b_spline import B_Spline
except ImportError:
    from b_spline import B_Spline

class KAN_layer(torch.nn.Module):
    """define a KAN layer using splines"""
    def __init__(self, k: int, input_dimensions: int, output_dimensions: int, inner_nodes: int, num=5, grid_eps=0.02, grid_range=[-1, 1], grid_extension=True, noise_scale=0.5, base_function=torch.nn.SiLU(), scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0) -> None:
        """
        Initialize the KAN layer.
        """
        super().__init__()
        self.k = k
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.inner_nodes = inner_nodes
        self.num = num
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.grid_extension = grid_extension
        self.mask = torch.ones(input_dimensions, output_dimensions)
        
        # Create grid with proper dimensions
        self.grid = torch.linspace(grid_range[0], grid_range[1], steps=self.num + 1)[None,:].expand(self.input_dimensions, self.num+1)
        
        # Create B_Spline instance
        self.b_spline = B_Spline(k=k, grid=self.grid, grid_extension=self.grid_extension)
        
        # Apply grid extension if requested
        if self.grid_extension:
            self.grid = self.b_spline.grid_extension(self.grid)
        
        # Initialize B-spline coefficients directly with proper random initialization
        # Calculate number of coefficients needed
        grid_size = self.grid.shape[1]
        n_coef = max(1, grid_size - k - 1)
        
        # Direct random initialization
        self.b_spline_basis_coefficents = torch.nn.Parameter(
            torch.randn(self.input_dimensions, self.output_dimensions, n_coef) * 0.1
        )
        
        self.b_spline_weight_coefficents = torch.nn.Parameter(torch.randn(input_dimensions, output_dimensions))
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(input_dimensions) + \
                         scale_base_sigma * (torch.rand(input_dimensions, output_dimensions)*2-1) * 1/np.sqrt(input_dimensions))
        self.scale_spline = torch.nn.Parameter(torch.ones(input_dimensions, output_dimensions) * scale_sp * 1 / np.sqrt(input_dimensions) * self.mask)
        self.base_function = base_function


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN layer.
        Each input goes through: w_base*base(x) + w_spline*spline(x)
        Then sum across input dimensions for each output node.
        """
        batch = x.shape[0]
        
        # Base function computation
        base = self.base_function(x)  # (batch, input_dimensions)
        
        # B-spline computation - Fix: use instance method properly
        y = self.b_spline.compute_coefficients_for_b_spline(
            x=x, 
            k=self.k, 
            grid=self.grid, 
            coefficients=self.b_spline_basis_coefficents
        )
        
        # Combine base and spline with learned weights
        # Fix: Use correct variable name self.scale_spline
        combined = self.scale_base[None,:,:] * base[:,:,None] + self.scale_spline[None,:,:] * y
        combined = self.mask[None,:,:] * combined
        
        # Sum across input dimensions to get output
        output = torch.sum(combined, dim=1)  # (batch, output_dimensions)
        
        return output


''''
How to implement a KAN layer:
What we need to intialize:
    - k: the order of the B-Spline
    - input_dimensions: the number of input dimensions
    - output_dimensions: the number of output dimensions
    - inner_nodes: the number of inner nodes
    - num: the number of grid points
    - grid_eps: the epsilon for the grid
    - grid_range: the range for the grid
    - grid_extension: whether to extend the grid
    - use a base function along with splines
    - use a spline activation
    - store parameters
    - heres how a kan works:
        -for each input, run it through the function: w_base*base(x) + w_spline*spline(x)
        -splines have a grid and trainable coefficents to allow them to form curves
        -the weights are trainable and can be used to control the contribution of the base function and the spline
        -Then for each node in the layer, sum up the outputs of each input once it has gone through the function
        -this is the output of the layer
        -continue for all nodes in the layer
        -then output those nodes and treat that as the input for the next layer
        -repeat for all layers
        
    -other key ideas:
        - increase grid size to increase the resolution of the spline during training

pytorch implementation:
plan: 
- create a class for the KAN layer that inherits from torch.nn.Module and the b spline class

'''