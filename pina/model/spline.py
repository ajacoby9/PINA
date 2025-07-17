"""Module for the Spline model class."""

import torch
from ..utils import check_consistency


class Spline(torch.nn.Module):
    """
    Spline model class.
    """

    def __init__(self, order=4, knots=None, control_points=None, grid_extension=True) -> None:
        """
        Initialization of the :class:`Spline` class.

        :param int order: The order of the spline. Default is ``4``.
        :param torch.Tensor knots: The tensor representing knots. If ``None``,
            the knots will be initialized automatically. Default is ``None``.
        :param torch.Tensor control_points: The control points. Default is
            ``None``.
        :raises ValueError: If the order is negative.
        :raises ValueError: If both knots and control points are ``None``.
        :raises ValueError: If the knot tensor is not one-dimensional.
        """
        super().__init__()

        check_consistency(order, int)

        if order < 0:
            raise ValueError("Spline order cannot be negative.")
        if knots is None and control_points is None:
            raise ValueError("Knots and control points cannot be both None.")

        self.order = order
        self.k = order - 1
        self.grid_extension = grid_extension

        if knots is not None and control_points is not None:
            self.knots = knots
            self.control_points = control_points

        elif knots is not None:
            print("Warning: control points will be initialized automatically.")
            print("         experimental feature")

            self.knots = knots
            n = len(knots) - order
            self.control_points = torch.nn.Parameter(
                torch.zeros(n), requires_grad=True
            )

        elif control_points is not None:
            print("Warning: knots will be initialized automatically.")
            print("         experimental feature")

            self.control_points = control_points

            n = len(self.control_points) - 1
            self.knots = {
                "type": "auto",
                "min": 0,
                "max": 1,
                "n": n + 2 + self.order,
            }

        else:
            raise ValueError("Knots and control points cannot be both None.")

        if self.knots.ndim > 2:
            raise ValueError("Knot vector must be one or two-dimensional.")

    def _create_basis(self, x, k, knots):
        """
        Compute the B-spline basis functions using recursion, aligned with pykan's B_batch.
        """
        if k == 0:
            # Base case: B-spline of order 0 is a step function
            value = (x[..., None] >= knots[..., :-1]) & (x[..., None] < knots[..., 1:])
            return value.to(x.dtype)

        # Recursive step
        # First term
        basis_k_minus_1_first = self._create_basis(x, k - 1, knots)
        
        denom1 = knots[..., k:-1] - knots[..., :-(k+1)]
        denom1 = torch.where(torch.abs(denom1) < 1e-8, torch.ones_like(denom1), denom1)
        
        numer1 = x[..., None] - knots[..., :-(k+1)]
        term1 = (numer1 / denom1) * basis_k_minus_1_first

        # Second term
        basis_k_minus_1_second = self._create_basis(x, k - 1, torch.roll(knots, shifts=-1, dims=-1))
        
        denom2 = knots[..., k+1:] - knots[..., 1:-k]
        denom2 = torch.where(torch.abs(denom2) < 1e-8, torch.ones_like(denom2), denom2)

        numer2 = knots[..., k+1:] - x[..., None]
        term2 = (numer2 / denom2) * basis_k_minus_1_second

        return term1 + term2


    def compute_control_points(self, x_eval, y_eval, new_knots):
        """
        Compute control points from given evaluations using least squares with regularization.
        """
        print("--- spline.compute_control_points ---")
        print(f"x_eval shape: {x_eval.shape}, range: [{x_eval.min():.4f}, {x_eval.max():.4f}]")
        print(f"y_eval shape: {y_eval.shape}, range: [{y_eval.min():.4f}, {y_eval.max():.4f}]")
        print(f"new_knots shape: {new_knots.shape}, range: [{new_knots.min():.4f}, {new_knots.max():.4f}]")
        
        A = self._create_basis(x_eval, self.k, new_knots)
        print(f"Basis matrix A shape: {A.shape}")
        
        in_dim = A.shape[1]
        out_dim = y_eval.shape[2]
        n_basis = A.shape[2]
        c = torch.zeros(in_dim, out_dim, n_basis).to(A.device)

        for i in range(in_dim):
            A_i = A[:, i, :]
            y_i = y_eval[:, i, :]
            
            # Regularized least squares
            A_t_A = A_i.T @ A_i
            A_t_y = A_i.T @ y_i
            ridge = 1e-6 * torch.eye(A_t_A.shape[0], device=A_i.device)
            
            try:
                c_i = torch.linalg.solve(A_t_A + ridge, A_t_y).T
                c[i, :, :] = c_i
            except torch.linalg.LinAlgError as e:
                print(f"ERROR: torch.linalg.solve failed for input_dim {i}: {e}")
        
        print(f"Computed control points shape: {c.shape}, range: [{c.min():.4f}, {c.max():.4f}]")
        self.knots = new_knots
        self.control_points = torch.nn.Parameter(c)
        print("--- End spline.compute_control_points ---\n")

    @property
    def control_points(self):
        """
        The control points of the spline.

        :return: The control points.
        :rtype: torch.Tensor
        """
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        """
        Set the control points of the spline.

        :param value: The control points.
        :type value: torch.Tensor | dict
        :raises ValueError: If invalid value is passed.
        """
        if isinstance(value, dict):
            if "n" not in value:
                raise ValueError("Invalid value for control_points")
            n = value["n"]
            dim = value.get("dim", 1)
            value = torch.zeros(n, dim)

        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
            
        if not isinstance(value, torch.Tensor):
            raise ValueError("Invalid value for control_points")
        self._control_points = value

    @property
    def knots(self):
        """
        The knots of the spline.

        :return: The knots.
        :rtype: torch.Tensor
        """
        return self._knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline.

        :param value: The knots.
        :type value: torch.Tensor | dict
        :raises ValueError: If invalid value is passed.
        """
        if isinstance(value, dict):

            type_ = value.get("type", "auto")
            min_ = value.get("min", 0)
            max_ = value.get("max", 1)
            n = value.get("n", 10)

            if type_ == "uniform":
                value = torch.linspace(min_, max_, n + self.k + 1)
            elif type_ == "auto":
                initial_knots = torch.ones(self.order + 1) * min_
                final_knots = torch.ones(self.order + 1) * max_

                if n < self.order + 1:
                    value = torch.concatenate((initial_knots, final_knots))
                elif n - 2 * self.order + 1 == 1:
                    value = torch.Tensor([(max_ + min_) / 2])
                else:
                    value = torch.linspace(min_, max_, n - 2 * self.order - 1)

                value = torch.concatenate((initial_knots, value, final_knots))

        if not isinstance(value, torch.Tensor):
            raise ValueError("Invalid value for knots")

        self._knots = value
    def forward(self, x):
        """
        Forward pass for the :class:`Spline` model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        t = self.knots
        k = self.k
        c = self.control_points

        # Create the basis functions
        # B will have shape (batch, in_dim, n_basis)
        B = self._create_basis(x, k, t)

        # KAN case where control points are (in_dim, out_dim, n_basis)
        if c.ndim == 3:
            y_ij = torch.einsum("bil,iol->bio", B, c)  # (batch, in_dim, out_dim)
            # sum over input dimensions
            y = torch.sum(y_ij, dim=1)  # (batch, out_dim)
        # Original test case
        else:
            B = B.squeeze(1)  # (batch, n_basis)
            if c.ndim == 1:
                y = torch.einsum("bi,i->b", B, c)
            else:
                y = torch.einsum("bi,ij->bj", B, c)

        return y

