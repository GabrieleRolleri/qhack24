import matplotlib.pyplot as plt
import torch
from qadence import QNN, QuantumCircuit, Z, add, chain, feature_map, hea
from qadence.types import BasisSet, ReuploadScaling


class Solver:
    criterion = torch.nn.MSELoss()

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 3,
        epochs: int = 500,
        points: int = 10,
        seed: int = 0,
    ) -> None:
        if n_qubits % 2 != 0:
            raise AttributeError("The number of qubits needs to be even.")
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_epochs = epochs
        self.points = points

    def calc_deriv(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute Laplassian of model that learns u(x,y), computes d^2u/dx^2+d^2u/dy^2
        using two evaluations of torch.autograd."""
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_y = grad[
            :, 1
        ]  # Select the column corresponding to the derivative with respect to y
        second_derivative_y = torch.autograd.grad(
            grad_y,
            inputs,
            torch.ones_like(grad_y),  # This tensor is required as grad_y is not scalar
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_x = grad[
            :, 0
        ]  # Select the column corresponding to the derivative with respect to y
        second_derivative_x = torch.autograd.grad(
            grad_x,
            inputs,
            torch.ones_like(grad_y),  # This tensor is required as grad_y is not scalar
            create_graph=True,
            retain_graph=True,
        )[0]
        lapl = torch.add(second_derivative_x[:, 0], second_derivative_y[:, 1])

        return lapl

    def loss_fn(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Loss function encoding the problem to solve."""
        # Equation loss
        model_output = model(inputs)
        deriv_model = self.calc_deriv(model_output, inputs)
        deriv_exact = torch.zeros_like(model_output[:, 0])

        ode_loss = self.criterion(deriv_model, deriv_exact)

        # Boundary loss, f(0) = 0
        boundary_left = inputs.detach().clone()
        boundary_left[:, 0] = 0
        boundary_bottom = inputs.detach().clone()
        boundary_bottom[:, 1] = 0
        boundary_right = inputs.detach().clone()
        boundary_right[:, 0] = 1
        boundary_top = inputs.detach().clone()
        boundary_top[:, 1] = 1

        boundary1_model = model(boundary_left)
        boundary1_exact = torch.sin(torch.pi * inputs[:, 1]).unsqueeze(1)
        boundary1_loss = self.criterion(boundary1_model, boundary1_exact)
        boundary2_model = model(boundary_bottom)
        boundary2_exact = torch.zeros_like(model_output)
        boundary2_loss = self.criterion(boundary2_model, boundary2_exact)
        boundary3_model = model(boundary_right)
        boundary3_exact = torch.exp(torch.tensor([-torch.pi])) * torch.sin(
            torch.pi * inputs[:, 1]
        ).unsqueeze(1)
        boundary3_loss = self.criterion(boundary3_model, boundary3_exact)
        boundary4_model = model(boundary_top)
        boundary4_exact = torch.zeros_like(model_output)
        boundary4_loss = self.criterion(boundary4_model, boundary4_exact)

        return (
            ode_loss + boundary1_loss + boundary2_loss + boundary3_loss + boundary4_loss
        )

    def train(self) -> None:
        # Feature map
        fm_x = feature_map(
            n_qubits=self.n_qubits // 2,
            support=torch.arange(0, self.n_qubits // 2),
            param="x",
            fm_type=BasisSet.FOURIER,
            reupload_scaling=ReuploadScaling.TOWER,
        )

        fm_y = feature_map(
            n_qubits=self.n_qubits // 2,
            param="y",
            support=torch.arange(self.n_qubits // 2, self.n_qubits),
            fm_type=BasisSet.FOURIER,
            reupload_scaling=ReuploadScaling.TOWER,
        )

        # Ansatz
        ansatz = hea(self.n_qubits, self.depth)

        # Observable
        observable = add(Z(i) for i in range(self.n_qubits))

        circuit = QuantumCircuit(self.n_qubits, chain(fm_x, fm_y, ansatz))
        self.model = QNN(circuit=circuit, observable=observable, inputs=["x", "y"])

        xmin = 0
        xmax = 0.999

        ymin = 0
        ymax = 0.999

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Training data. We unsqueeze essentially making each batch have a single x value.
            xy_train = (
                torch.stack(
                    (
                        xmin
                        + (xmax - xmin)
                        * torch.rand(self.points, self.points, requires_grad=True),
                        ymin
                        + (ymax - ymin)
                        * torch.rand(self.points, self.points, requires_grad=True),
                    )
                )
                .swapdims(0, 2)
                .flatten(end_dim=1)
            )

            loss = self.loss_fn(inputs=xy_train, model=self.model)
            if epoch % 50 == 0:
                print("Loss:", round(loss.item(), 4))

            loss.backward()
            optimizer.step()

    def plot(self) -> None:
        xmin = 0
        xmax = 0.999

        # result_exact = f_exact(x_test).flatten()
        x = torch.arange(xmin, xmax, 0.01)
        xy_test = torch.cartesian_prod(x, x)

        X, Y = torch.meshgrid(x, x)

        result_model = (
            self.model(xy_test).detach().unflatten(0, (x.shape[0], x.shape[0]))
        )

        # plt.plot(x_test, result_exact, label = "Exact solution")
        plt.pcolormesh(
            X.detach().numpy(),
            Y.detach().numpy(),
            result_model.squeeze(2).detach().numpy(),
            label=" Trained model",
        )
        plt.show()
