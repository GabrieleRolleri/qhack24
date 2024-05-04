from itertools import product
from Solver import Solver
import torch

n_points_1d = 100  # Use a grid of 100x100
scores = []

# Create a length = 10.000 tensor of the (x, y) coordinates between 0 and 1
domain_1d = torch.linspace(0, 1.0, steps=n_points_1d)
domain = torch.tensor(list(product(domain_1d, domain_1d)))


# The exact solution for the Laplace equation
def exact_laplace(domain: torch.Tensor):
    exp_x = torch.exp(-torch.pi * domain[:, 0])
    sin_y = torch.sin(torch.pi * domain[:, 1])
    return exp_x * sin_y

for seed in range(1,11):
    s=Solver(seed=seed)
    s.train()
    # Getting the exact solution and the DQC solution
    exact_sol = exact_laplace(domain).reshape(n_points_1d, n_points_1d).T
    dqc_sol = s.model(domain).reshape(n_points_1d, n_points_1d).T.detach()
    # Mean Squared Error as the comparison criterion
    criterion = torch.nn.MSELoss()
    # Final score, the lower the better
    scores.append(criterion(dqc_sol, exact_sol))

print(scores)