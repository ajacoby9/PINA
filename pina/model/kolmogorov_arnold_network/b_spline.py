"""Module for the B-Spline model class."""
import torch

class B_Spline:
    """
    Define the B-Spline model class. Create the infastructure for fast and efficent B-Spline operations 
    and construction in order to provide a basis for Komogorov Arnold Networks.
    """
    def __init__(self, k: int, grid: torch.Tensor, grid_extension: bool = True, x: torch.Tensor = None):
        """
        Initialize the B-Spline class.
        """
        self.k = k
        self.k_extend = k  # For grid extension
        self.grid = grid
        self.grid_extension = grid_extension
        self.x = x

    def create_basis_functions(self, k: int, grid: torch.Tensor, x: torch.Tensor):
        """
        Create the basis functions for the B-Spline.
        """
        x = x.unsqueeze(dim=2)
        grid = grid.unsqueeze(dim=0)
        if k == 0:
            return (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        else:
            basis_function = self.create_basis_functions(k - 1, grid, x)
            return (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * basis_function[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * basis_function[:, :, 1:]

    def compute_coefficients_for_b_spline(self, x: torch.Tensor, k: int, grid: torch.Tensor, coefficients: torch.Tensor):
        """
        use the basis function to compute the coefficients
        """
        b_splines = self.create_basis_functions(k, grid, x)
        y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coefficients.to(b_splines.device))
        return y_eval
    
    def compute_coefficients_from_b_spline(self, x_eval, y_eval, grid, k):
        batch = x_eval.shape[0]
        in_dim = x_eval.shape[1]
        out_dim = y_eval.shape[2]
        n_coef = grid.shape[1] - k - 1
        mat = self.create_basis_functions(k, grid, x_eval)
        mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
        y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
        device = mat.device
        try:
            coefficients = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
        except:
            print('lstsq failed')
        return coefficients

    def grid_extension(self, grid):
        """
        Extend the grid by the order of the B-Spline.
        """
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(self.k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

        return grid
    
    def __call__(self, x):
        """Make B_Spline callable - placeholder for now"""
        # This needs proper implementation with coefficients
        return x  # Temporary placeholder

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