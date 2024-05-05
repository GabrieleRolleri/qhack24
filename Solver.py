import matplotlib.pyplot as plt
import torch
from qadence import (
    QNN,
    QuantumCircuit,
    Z,
    add,
    chain,
    feature_map,
    hea,
    kron,
    load,
    save,
)
from qadence.draw import display
from qadence.types import BasisSet, ReuploadScaling


class Solver:
    criterion = torch.nn.MSELoss()

    def __init__(
        self,
        n_qubits: int = 8,
        depth: int = 6,
        epochs: int = 800,
        points: int = 10,
        seed: int = 0,
        ode_rescale: float = 100,
    ) -> None:
        """Class for solving the 2D Laplace equation using a QNN.
        Arguments:
             - n_qubits: number of qubits to use for the quantum circuits (multiple of 2)
             - depth: the circuits depth for the HEA ansatz
             - epochs: number of training epochs for the QNN
             - points: number of square grid collocation points to train on
             - seed: specific seed value for the pyTorch PRNG
             - ode_rescale: factor by which to dampen the ODE loss"""
        if n_qubits % 2 != 0:
            raise AttributeError("The number of qubits needs to be even.")
        # Choses specific the randomness seed
        torch.manual_seed(seed)

        # Sets user-defined parameters as attributes
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_epochs = epochs
        self.points = points
        self.ode_rescale = ode_rescale

    def laplacian(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Returns Laplacian of model that learns u(x,y), computes d^2u/dx^2+d^2u/dy^2
        using two evaluations of torch.autograd."""

        # Calculates first order derivatives
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
        ]  # Select the column corresponding to the derivative with respect to x
        second_derivative_x = torch.autograd.grad(
            grad_x,
            inputs,
            torch.ones_like(grad_y),  # This tensor is required as grad_y is not scalar
            create_graph=True,
            retain_graph=True,
        )[0]

        return torch.add(second_derivative_x[:, 0], second_derivative_y[:, 1])

    def loss_fn(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Loss function encoding the problem to solve."""
        # Equation loss
        model_output = model(inputs)  # Evaluates model in training points
        deriv_model = self.laplacian(
            model_output, inputs
        )  # Calculates Laplacian for the trained model
        deriv_exact = torch.zeros_like(model_output[:, 0])

        ode_loss = self.criterion(deriv_model, deriv_exact)

        # Defining the  boundary collocation points
        boundary_left = inputs.detach().clone()
        boundary_left[:, 0] = 0
        boundary_bottom = inputs.detach().clone()
        boundary_bottom[:, 1] = 0
        boundary_right = inputs.detach().clone()
        boundary_right[:, 0] = 1
        boundary_top = inputs.detach().clone()
        boundary_top[:, 1] = 1

        # Evaluating in the boundaries and calculating the loss
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
            ode_loss / self.ode_rescale
            + boundary1_loss
            + boundary2_loss
            + boundary3_loss
            + boundary4_loss
        )

    def exact_laplace(self, domain: torch.Tensor):
        """Returns exact solution of the 2D Laplace equation"""
        exp_x = torch.exp(-torch.pi * domain[:, 0])
        sin_y = torch.sin(torch.pi * domain[:, 1])
        return exp_x * sin_y

    def train(self) -> None:
        """Trains the QNN and populates the model attribute with the resulting model"""
        # Create two feature maps (disjunct supports)
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

        # Use the canonical observable
        observable = add(Z(i) for i in range(self.n_qubits))

        circuit = QuantumCircuit(self.n_qubits, chain(kron(fm_x, fm_y), ansatz))
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
                print(f"Loss (epoch {epoch}/{self.n_epochs}):", round(loss.item(), 4))

            loss.backward()
            optimizer.step()

    def plot(self, filename: str = "") -> None:
        """Creates plot comparing the exact solution to the one of the trained model"""
        x = torch.arange(0, 1, 0.01)
        xy_test = torch.cartesian_prod(x, x)

        X, Y = torch.meshgrid(x * 100, x * 100)

        result_model = (
            self.model(xy_test).detach().unflatten(0, (x.shape[0], x.shape[0]))
        )
        result_exact = self.exact_laplace(xy_test).flatten()

        fig, axs = plt.subplots(1, 2)
        axs[0].pcolormesh(
            X.detach().numpy(),
            Y.detach().numpy(),
            result_model.squeeze(2).detach().numpy(),
            label=" Trained model",
        )
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].title.set_text("DQC solution for u(x,y)")
        axs[1].pcolormesh(
            X.detach().numpy(),
            Y.detach().numpy(),
            result_exact.reshape(100, 100),
            label=" Trained model",
        )
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        axs[1].title.set_text("Analytical solution for u(x,y)")
        fig.suptitle(
            f"Model parameters: qubits={self.n_qubits}, epochs={self.n_epochs} and depth={self.depth}"
        )
        if filename != "":
            plt.savefig(filename, dpi=320)
        else:
            plt.show()

    def visualize(self) -> None:
        """Creates a circuit visualisation using qadence visualise."""
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

        display(
            QuantumCircuit(self.n_qubits, chain(kron(fm_x, fm_y), ansatz, observable))
        )

    def save(self, filename: str) -> None:
        if hasattr(self, "model"):
            save(self.model, folder=".", file_name=filename, format="PT")

    def load(self, filename: str) -> None:
        self.model = load(filename)
