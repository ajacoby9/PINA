"""Create the infastructure for a KAN layer"""
import torch
from b_spline import B_Spline
class KAN_layer(B_Spline):
    """define a KAN layer using splines"""
    def __init__(self, k: int, grid: torch.Tensor, grid_extension: bool, input_dimensions: int, output_dimensions: int, inner_nodes: int) -> None:
        """
        Initialize the KAN layer.
        """
        super().__init__()
        self.order = k
        self.grid = grid
        self.grid_extension = grid_extension
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.inner_nodes = inner_nodes
        self.spline = B_Spline(self.k, self.knots, grid)
        self.weights_spline_activation = torch.nn.Parameter(torch.randn(1, 1, 1, 1))
        self.weights_sigmoid_activation = torch.nn.Parameter(torch.randn(1, 1, 1, 1))

    def activation_function(self, x: torch.Tensor, ) -> torch.Tensor:
        """Define the activation function in the KAN layer"""
        return self.weights_sigmoid_activation * torch.silu(x) + self.weights_spline_activation * self.spline(x)
    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        """
        first, take each of the current values of x, run them thorough 
        the function (kinda an activation function but not really because this isnt an MLP)
        then take those values and sum them for each of the new x values post layer: (fully connected right now)
        to do: add sparsificaiton, (could be done through training potentually? maybe addition to the cost function)
        
        """
        pass
    