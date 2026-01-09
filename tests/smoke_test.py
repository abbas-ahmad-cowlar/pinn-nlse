"""
Smoke Test for PINN-NLSE Project
Verifies: NumPy FFT, PyTorch autograd, sech energy, and complex arithmetic.
"""
import numpy as np
import torch

# Test 1: NumPy FFT roundtrip
N = 1024
tau = np.linspace(-20, 20, N, endpoint=False)
dtau = tau[1] - tau[0]
sech = 1.0 / np.cosh(tau)
spectrum = np.fft.fft(sech)
reconstructed = np.fft.ifft(spectrum)
assert np.max(np.abs(reconstructed - sech)) < 1e-12, "FFT roundtrip failed"

# Test 2: sech energy = 2.0
energy = np.sum(np.abs(sech) ** 2) * dtau
assert abs(energy - 2.0) < 0.01, f"int sech^2 dtau = {energy}, expected 2.0"

# Test 3: PyTorch autograd (core PINN requirement)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sin(x)
dy_dx = torch.autograd.grad(
    y, x, grad_outputs=torch.ones_like(y), create_graph=True
)[0]
expected = torch.cos(x)
assert torch.allclose(dy_dx, expected, atol=1e-6), "Autograd d(sin)/dx != cos"

# Test 4: Second derivative via autograd
d2y_dx2 = torch.autograd.grad(
    dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True
)[0]
expected2 = -torch.sin(x)
assert torch.allclose(d2y_dx2, expected2, atol=1e-6), "Autograd d2(sin)/dx2 != -sin"

# Test 5: Complex arithmetic in PyTorch
u_re = torch.tensor([1.0, 0.0])
u_im = torch.tensor([0.0, 1.0])
intensity = u_re ** 2 + u_im ** 2
assert torch.allclose(intensity, torch.ones(2)), "|u|^2 computation failed"

# Test 6: CUDA availability (informational)
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"

print("All smoke tests passed.")
print(f"   int sech^2(tau) dtau = {energy:.6f} (expected: 2.0)")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {cuda_available} ({device_name})")
print(f"   Autograd 1st derivative: OK")
print(f"   Autograd 2nd derivative: OK")