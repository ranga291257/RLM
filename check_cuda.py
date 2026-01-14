import torch
print("PyTorch Version: ", torch.__version__)

print("GPU available: ", torch.cuda.is_available())

def check_cuda():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA Device Name: {device_name}")
        
        # Get device properties
        device_properties = torch.cuda.get_device_properties(current_device)
        print(f"Total Memory: {device_properties.total_memory / 1024**3:.2f} GB")
        print(f"GPU Name: {device_properties.name}")
        print(f"Multi Processor Count: {device_properties.multi_processor_count}")
        
        # Test CUDA tensor creation
        print("\nTesting CUDA Tensor Creation:")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"Test Tensor Device: {x.device}")

if __name__ == "__main__":
    check_cuda()