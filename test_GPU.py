import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    A = torch.randn(1000, 1000, device="cuda")
    B = torch.randn(1000, 1000, device="cuda")

    C = A @ B
    torch.cuda.synchronize()

    print("Result shape:", C.shape)
    print("Device:", C.device)