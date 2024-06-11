import torch
import time


def perform_matrix_multiplication(size=1000, device="cpu", print_time=True):
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Measure execution time
    start_time = time.time()
    c = torch.matmul(a, b)
    end_time = time.time()

    elapsed_time = end_time - start_time
    if print_time:
        print(f"{device.upper()} execution time: {elapsed_time:.6f} seconds")
    return elapsed_time


if __name__ == "__main__":
    # Perform and time matrix multiplication on CPU
    cpu_time = perform_matrix_multiplication(size=2000, device="cpu")

    # Check if MPS is available and perform matrix multiplication
    if torch.backends.mps.is_available():
        mps_time = perform_matrix_multiplication(size=2000, device="mps")
    else:
        print("MPS is not available on this Mac M1 system.")
