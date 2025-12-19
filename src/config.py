"""
Configuration Module - Physical & PINN Hyperparameters
=======================================================

Centralizes ALL parameters for the PINN-NLSE project.
Imported by every other module and notebook.

Sign Convention: Agrawal (Nonlinear Fiber Optics, 6th ed.)
    s = -sign(beta_2) = +1 for anomalous dispersion (soliton regime)
    Normalized NLSE: i d_xi u + (s/2) d_tau^2 u + N^2 |u|^2 u = 0

"""