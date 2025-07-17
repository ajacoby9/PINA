"""Create the infrastructure for a KAN layer"""
import torch
import numpy as np

from pina.model.spline import Spline


class KAN_layer(torch.nn.Module):
    """define a KAN layer using splines"""
    def __init__(self, k: int, input_dimensions: int, output_dimensions: int, inner_nodes: int, num=3, grid_eps=0.02, grid_range=[-1, 1], grid_extension=True, noise_scale=0.1, base_function=torch.nn.SiLU(), scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, sparse_init=True, sp_trainable=True, sb_trainable=True) -> None:
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
        
        if sparse_init:
            self.mask = torch.nn.Parameter(self.sparse_mask(input_dimensions, output_dimensions)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(input_dimensions, output_dimensions)).requires_grad_(False)        
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=self.num + 1)[None,:].expand(self.input_dimensions, self.num+1)
        
        if grid_extension:
            h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
            for i in range(self.k):
                grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
                grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        
        n_coef = grid.shape[1] - (self.k + 1)
        
        control_points = torch.nn.Parameter(
            torch.randn(self.input_dimensions, self.output_dimensions, n_coef) * noise_scale
        )

        self.spline = Spline(order=self.k+1, knots=grid, control_points=control_points, grid_extension=grid_extension)

        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(input_dimensions) + \
                         scale_base_sigma * (torch.rand(input_dimensions, output_dimensions)*2-1) * 1/np.sqrt(input_dimensions), requires_grad=sb_trainable)
        self.scale_spline = torch.nn.Parameter(torch.ones(input_dimensions, output_dimensions) * scale_sp * 1 / np.sqrt(input_dimensions) * self.mask, requires_grad=sp_trainable)
        self.base_function = base_function

    @staticmethod
    def sparse_mask(in_dimensions: int, out_dimensions: int) -> torch.Tensor:
        '''
        get sparse mask
        '''
        in_coord = torch.arange(in_dimensions) * 1/in_dimensions + 1/(2*in_dimensions)
        out_coord = torch.arange(out_dimensions) * 1/out_dimensions + 1/(2*out_dimensions)

        dist_mat = torch.abs(out_coord[:,None] - in_coord[None,:])
        in_nearest = torch.argmin(dist_mat, dim=0)
        in_connection = torch.stack([torch.arange(in_dimensions), in_nearest]).permute(1,0)
        out_nearest = torch.argmin(dist_mat, dim=1)
        out_connection = torch.stack([out_nearest, torch.arange(out_dimensions)]).permute(1,0)
        all_connection = torch.cat([in_connection, out_connection], dim=0)
        mask = torch.zeros(in_dimensions, out_dimensions)
        mask[all_connection[:,0], all_connection[:,1]] = 1.
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the KAN layer.
        Each input goes through: w_base*base(x) + w_spline*spline(x)
        Then sum across input dimensions for each output node.
        """
        base = self.base_function(x)  # (batch, input_dimensions)
        
        basis = self.spline.basis(x, self.spline.k, self.spline.knots)
        spline_out_per_input = torch.einsum("bil,iol->bio", basis, self.spline.control_points)

        base_term = self.scale_base[None, :, :] * base[:, :, None]
        spline_term = self.scale_spline[None, :, :] * spline_out_per_input
        combined = base_term + spline_term
        combined = self.mask[None,:,:] * combined
        
        output = torch.sum(combined, dim=1)  # (batch, output_dimensions)
        
        return output

    def update_grid_from_samples(self, x: torch.Tensor, mode: str = 'sample'):
        """
        Update grid from input samples to better fit data distribution.
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            
            x_sorted = torch.sort(x, dim=0)[0]  # (batch_size, input_dimensions)
            
            basis = self.spline.basis(x_sorted, self.spline.k, self.spline.knots)
            y_eval = torch.einsum("bil,iol->bio", basis, self.spline.control_points)

            num_interval = self.num
            
            def get_grid_adaptive(num_intervals: int):
                """Create adaptive grid based on sample quantiles"""
                indices = [int(batch_size * i / num_intervals) for i in range(num_intervals)]
                indices.append(batch_size - 1) 
                
                grid_adaptive = x_sorted[indices, :].transpose(0, 1)  # (input_dimensions, num_intervals+1)
                
                margin = 0.01  # Small margin for numerical stability
                grid_min = grid_adaptive[:, [0]] - margin
                grid_max = grid_adaptive[:, [-1]] + margin
                h = (grid_max - grid_min) / num_intervals
                
                grid_uniform = grid_min + h * torch.arange(
                    num_intervals + 1, device=x.device, dtype=x.dtype
                )[None, :]
                
                grid_blended = (self.grid_eps * grid_uniform + 
                              (1 - self.grid_eps) * grid_adaptive)
                
                return grid_blended
            
            new_grid = get_grid_adaptive(num_interval)
            
            if mode == 'grid':
                sample_grid = get_grid_adaptive(2 * num_interval)
                x_eval = sample_grid.transpose(0, 1)  # (batch_size, input_dimensions)
                # Re-evaluate at denser grid
                basis = self.spline.basis(x_eval, self.spline.k, self.spline.knots)
                y_eval = torch.einsum("bil,iol->bio", basis, self.spline.control_points)

                x_sorted = x_eval
            
            if self.grid_extension:
                h = (new_grid[:, [-1]] - new_grid[:, [0]]) / (new_grid.shape[1] - 1)
                for i in range(self.k):
                    new_grid = torch.cat([new_grid[:, [0]] - h, new_grid], dim=1)
                    new_grid = torch.cat([new_grid, new_grid[:, [-1]] + h], dim=1)
            
            self.spline.knots = new_grid
            
            try:
                self.spline.compute_control_points(x_sorted, y_eval)
                    
            except Exception as e:
                # If coefficient fitting fails, keep old coefficients
                print(f"Warning: Failed to update coefficients during grid refinement: {e}")
                pass

    def update_grid_resolution(self, new_num: int):
        """
        Update grid resolution to a new number of intervals.
        """
        with torch.no_grad():
            # Sample the current spline function on a dense grid
            x_eval = torch.linspace(
                self.grid_range[0], 
                self.grid_range[1], 
                steps=2 * new_num, 
                device=self.spline.knots.device
            )
            x_eval = x_eval.unsqueeze(1).expand(-1, self.input_dimensions)

            basis = self.spline.basis(x_eval, self.spline.k, self.spline.knots)
            y_eval = torch.einsum("bil,iol->bio", basis, self.spline.control_points)

            # Update num and create a new grid
            self.num = new_num
            new_grid = torch.linspace(
                self.grid_range[0], 
                self.grid_range[1], 
                steps=self.num + 1, 
                device=self.spline.knots.device
            )
            new_grid = new_grid[None, :].expand(self.input_dimensions, self.num + 1)

            if self.grid_extension:
                h = (new_grid[:, [-1]] - new_grid[:, [0]]) / (new_grid.shape[1] - 1)
                for i in range(self.k):
                    new_grid = torch.cat([new_grid[:, [0]] - h, new_grid], dim=1)
                    new_grid = torch.cat([new_grid, new_grid[:, [-1]] + h], dim=1)
            
            # Update spline with the new grid and re-compute control points
            self.spline.knots = new_grid
            self.spline.compute_control_points(x_eval, y_eval)

    def get_grid_statistics(self):
        """Get statistics about the current grid for debugging/analysis"""
        return {
            'grid_shape': self.spline.knots.shape,
            'grid_min': self.spline.knots.min().item(),
            'grid_max': self.spline.knots.max().item(),
            'grid_range': (self.spline.knots.max() - self.spline.knots.min()).mean().item(),
            'num_intervals': self.spline.knots.shape[1] - 1 - (2*self.k if self.spline.grid_extension else 0)
        }