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
        self.use_grid_extension = grid_extension  # Renamed to avoid conflict with method
        self.x = x

    def create_basis_functions(self, k: int, grid: torch.Tensor, x: torch.Tensor):
        """
        Create the basis functions for the B-Spline.
        Fixed tensor dimension handling for proper B-spline computation.
        """
        # Ensure proper dimensions
        if x.dim() == 2:
            x = x.unsqueeze(dim=2)  # (batch, input_dim, 1)
        if grid.dim() == 2:
            grid = grid.unsqueeze(dim=0)  # (1, input_dim, grid_points)
        
        # Base case: k=0
        if k == 0:
            # Check if we have enough grid points
            if grid.shape[2] <= 1:
                return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
            return ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        
        # Recursive case: k > 0
        # Get basis functions for k-1
        basis_prev = self.create_basis_functions(k - 1, grid, x)
        
        # Check if we have enough coefficients for the computation
        if basis_prev.shape[2] == 0:
            return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
        
        # Compute the two terms of the B-spline recurrence relation
        # First term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
        # Second term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        
        n_basis = basis_prev.shape[2]
        if n_basis == 0:
            return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
        
        # Ensure we have enough grid points for the computation
        grid_size = grid.shape[2]
        if grid_size < k + 2:
            return torch.zeros(x.shape[0], x.shape[1], max(0, grid_size - k - 1), device=x.device)
        
        # Calculate indices for slicing
        max_basis_funcs = max(0, grid_size - k - 1)
        
        if max_basis_funcs == 0:
            return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)
        
        # Initialize output
        result = torch.zeros(x.shape[0], x.shape[1], max_basis_funcs, device=x.device)
        
        # First term computation where possible
        if n_basis > 0 and max_basis_funcs > 0:
            end_idx = min(n_basis, max_basis_funcs)
            
            # Denominators for first term
            denom1 = grid[:, :, k:k+end_idx] - grid[:, :, :end_idx]
            # Avoid division by zero
            denom1 = torch.where(torch.abs(denom1) < 1e-8, torch.ones_like(denom1), denom1)
            
            # Numerators for first term
            numer1 = x - grid[:, :, :end_idx]
            
            # First term contribution
            if end_idx <= basis_prev.shape[2]:
                result[:, :, :end_idx] += (numer1 / denom1) * basis_prev[:, :, :end_idx]
        
        # Second term computation where possible  
        if n_basis > 1 and max_basis_funcs > 0:
            end_idx = min(n_basis - 1, max_basis_funcs)
            
            # Denominators for second term
            denom2 = grid[:, :, k+1:k+1+end_idx] - grid[:, :, 1:1+end_idx]
            # Avoid division by zero
            denom2 = torch.where(torch.abs(denom2) < 1e-8, torch.ones_like(denom2), denom2)
            
            # Numerators for second term
            numer2 = grid[:, :, k+1:k+1+end_idx] - x
            
            # Second term contribution
            if end_idx <= basis_prev.shape[2] - 1:
                result[:, :, :end_idx] += (numer2 / denom2) * basis_prev[:, :, 1:1+end_idx]
        
        return result

    def compute_coefficients_for_b_spline(self, x: torch.Tensor, k: int, grid: torch.Tensor, coefficients: torch.Tensor):
        """
        use the basis function to compute the coefficients
        """
        b_splines = self.create_basis_functions(k, grid, x)
        
        # Handle empty basis case
        if b_splines.shape[2] == 0:
            return torch.zeros(x.shape[0], x.shape[1], coefficients.shape[1], device=x.device)
        
        # Ensure coefficient dimensions match
        coef_device = coefficients.to(b_splines.device)
        
        # Adjust coefficients size if needed
        if coef_device.shape[2] != b_splines.shape[2]:
            min_size = min(coef_device.shape[2], b_splines.shape[2])
            coef_device = coef_device[:, :, :min_size]
            b_splines = b_splines[:, :, :min_size]
        
        y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef_device)
        return y_eval
    
    def compute_coefficients_from_b_spline(self, x_eval, y_eval, grid, k):
        """
        Compute coefficients from given B-spline evaluations using least squares fitting.
        This method fits B-spline coefficients to match given target values.
        """
        batch = x_eval.shape[0]
        in_dim = x_eval.shape[1]
        out_dim = y_eval.shape[2]
        n_coef = max(1, grid.shape[1] - k - 1)
        
        if n_coef <= 0:
            return torch.zeros(in_dim, out_dim, 1, device=grid.device)
        
        # Get basis function matrix
        mat = self.create_basis_functions(k, grid, x_eval)  # (batch, in_dim, n_basis)
        
        if mat.shape[2] == 0:
            return torch.zeros(in_dim, out_dim, 1, device=grid.device)
        
        device = mat.device
        
        # Solve least squares problem for each input-output pair separately
        coefficients = torch.zeros(in_dim, out_dim, mat.shape[2], device=device)
        
        for i in range(in_dim):
            for j in range(out_dim):
                # Extract data for this input-output pair
                A = mat[:, i, :]  # (batch, n_basis)
                b = y_eval[:, i, j]  # (batch,)
                
                # Solve least squares: A @ c = b
                try:
                    # Use pseudoinverse for robust solving
                    if A.shape[0] >= A.shape[1]:  # Overdetermined or square system
                        c = torch.linalg.lstsq(A, b).solution
                    else:  # Underdetermined system
                        c = torch.linalg.pinv(A) @ b
                    
                    coefficients[i, j, :c.shape[0]] = c
                    
                except Exception as e:
                    return BrokenPipeError("lstsq failed")
                    # If lstsq fails, use small random coefficients
                    #coefficients[i, j, :] = torch.randn(mat.shape[2], device=device) * 0.01
        
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
    
    def __call__(self, x, coefficients):
        """Evaluate B-spline with given coefficients"""
        return self.compute_coefficients_for_b_spline(x, self.k, self.grid, coefficients)

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