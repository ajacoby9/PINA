"""Module for the B-Spline model class."""
import torch
from ...utils import check_consistency

class B_Spline:
    """
    Define the B-Spline model class. Create the infastructure for fast and efficent B-Spline operations 
    and construction in order to provide a basis for Komogorov Arnold Networks.
    """
    def __init__(self, k: int, grid: torch.Tensor, grid_extension: bool = True, x: torch.Tensor = None):
        """
        Inilialize the B-Spline class.
        """
        self.k = k
        self.grid = grid
        self.grid_extension = grid_extension
        self.x = x

    def create_basis_functions(self, k: int, grid: torch.Tensor, x: torch.Tensor):
        """
        Create the basis functions for the B-Spline.
        """
        if k == 0:
            return (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        else:
            basis_function = self.create_basis_functions(k - 1, grid, x)
            return (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * basis_function[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * basis_function[:, :, 1:]

    def compute_coefficients(self, x: torch.Tensor, k: int, grid: torch.Tensor, coefficients: torch.Tensor):
        """
        use the basis function to compute the coefficients
        """
        b_splines = self.create_basis_functions(k, grid, x)
        y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coefficients.to(b_splines.device))
        return y_eval
    

    def grid_extension(self, grid):
        pass

"""
Plan:
1) define B-splines:
-grid of knots
-create basis functions
-find coefficents
-grid extension
-compute output of B-spline
-derivative with autograd i think
2) define KAN layer:
-allow for inputted input size, output size
-inherit from B-spline class B splines
-for each node in the layer, create a B-spline
-for each node in the layer also create a silu layer
-create a weight matrix for the B-splines
-create a weight matrix for the silu layers
for each node the activation function is:
-w_spline * B-spline(x) + w_silu * silu(x)
3) define KAN feed forward+ training:
-define a forward pass
-define a loss function
-define a training loop
-define a test loop
-define a save model function
-define a load model function
"""