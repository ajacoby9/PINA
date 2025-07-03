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

    def update_grid_from_samples(self, x: torch.Tensor, mode: str = 'sample'):
        """
        Update grid from input samples to better fit data distribution.
        
        Args:
            x: Input samples, shape (batch_size, input_dimensions)
            mode: 'sample' or 'grid' - determines sampling strategy
            
        Returns:
            None
        
        Example:
            >>> layer = KAN_layer(k=3, input_dimensions=2, output_dimensions=1, inner_nodes=0, num=5)
            >>> x = torch.randn(100, 2)  # 100 samples, 2 input dimensions
            >>> layer.update_grid_from_samples(x)
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            
            # Sort input samples along batch dimension for each input dimension
            x_sorted = torch.sort(x, dim=0)[0]  # (batch_size, input_dimensions)
            
            # Evaluate current spline at sorted positions
            y_eval = self.b_spline.compute_coefficients_for_b_spline(
                x=x_sorted,
                k=self.k,
                grid=self.grid,
                coefficients=self.b_spline_basis_coefficents
            )
            
            # Calculate number of intervals (excluding extended grid points)
            if self.grid_extension:
                # Account for extended grid points (k points on each side)
                num_interval = self.grid.shape[1] - 1 - 2*self.k
            else:
                num_interval = self.grid.shape[1] - 1
            
            num_interval = max(1, num_interval)  # Ensure at least 1 interval
            
            def get_grid_adaptive(num_intervals: int):
                """Create adaptive grid based on sample quantiles"""
                # Calculate quantile positions for grid points
                indices = [int(batch_size * i / num_intervals) for i in range(num_intervals)]
                indices.append(batch_size - 1)  # Last sample
                
                # Get adaptive grid points from sorted samples
                grid_adaptive = x_sorted[indices, :].transpose(0, 1)  # (input_dimensions, num_intervals+1)
                
                # Create uniform grid as baseline
                margin = 0.01  # Small margin for numerical stability
                grid_min = grid_adaptive[:, [0]] - margin
                grid_max = grid_adaptive[:, [-1]] + margin
                h = (grid_max - grid_min) / num_intervals
                
                grid_uniform = grid_min + h * torch.arange(
                    num_intervals + 1, device=x.device, dtype=x.dtype
                )[None, :]
                
                # Blend adaptive and uniform grids
                grid_blended = (self.grid_eps * grid_uniform + 
                              (1 - self.grid_eps) * grid_adaptive)
                
                return grid_blended
            
            # Create new grid
            new_grid = get_grid_adaptive(num_interval)
            
            # If mode is 'grid', use denser sampling for coefficient fitting
            if mode == 'grid':
                sample_grid = get_grid_adaptive(2 * num_interval)
                x_eval = sample_grid.transpose(0, 1)  # (batch_size, input_dimensions)
                # Re-evaluate at denser grid
                y_eval = self.b_spline.compute_coefficients_for_b_spline(
                    x=x_eval,
                    k=self.k,
                    grid=self.grid,
                    coefficients=self.b_spline_basis_coefficents
                )
                x_sorted = x_eval
            
            # Apply grid extension if needed
            if self.grid_extension:
                new_grid = self.b_spline.grid_extension(new_grid)
            
            # Update grid
            self.grid = new_grid
            
            # Recompute B-spline coefficients to maintain continuity
            try:
                new_coefficients = self.b_spline.compute_coefficients_from_b_spline(
                    x_eval=x_sorted,
                    y_eval=y_eval,
                    grid=self.grid,
                    k=self.k
                )
                
                # Update coefficients with proper shape handling
                if new_coefficients.shape == self.b_spline_basis_coefficents.shape:
                    self.b_spline_basis_coefficents.data = new_coefficients
                else:
                    # Handle size mismatch by creating new parameter
                    self.b_spline_basis_coefficents = torch.nn.Parameter(new_coefficients)
                    
            except Exception as e:
                # If coefficient fitting fails, keep old coefficients
                print(f"Warning: Failed to update coefficients during grid refinement: {e}")
                # Optionally, we could reset coefficients to small random values
                pass

    def get_grid_statistics(self):
        """Get statistics about the current grid for debugging/analysis"""
        return {
            'grid_shape': self.grid.shape,
            'grid_min': self.grid.min().item(),
            'grid_max': self.grid.max().item(),
            'grid_range': (self.grid.max() - self.grid.min()).mean().item(),
            'num_intervals': self.grid.shape[1] - 1 - (2*self.k if self.grid_extension else 0)
        }

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