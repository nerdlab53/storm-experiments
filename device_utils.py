"""
Device utilities for automatic device detection and management
Supports CUDA, MPS (Apple Silicon), and CPU
"""

import torch
import warnings

def get_device():
    """
    Get the best available device for training
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_type():
    """
    Get device type string for autocast
    """
    device = get_device()
    if device.type == 'cuda':
        return 'cuda'
    elif device.type == 'mps':
        return 'cpu'  # MPS doesn't support autocast, use CPU autocast
    else:
        return 'cpu'

def get_autocast_dtype(device_type=None):
    """
    Get appropriate dtype for autocast based on device
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'cuda':
        return torch.bfloat16
    else:
        # MPS and CPU work better with float16 or disabled autocast
        return torch.float16

def is_autocast_enabled(device_type=None):
    """
    Check if autocast should be enabled for the device
    """
    if device_type is None:
        device_type = get_device_type()
    
    # Enable autocast for CUDA, disable for MPS (stability)
    return device_type == 'cuda'

def move_to_device(tensor_or_module, device=None):
    """
    Move tensor or module to device with proper error handling
    """
    if device is None:
        device = get_device()
    
    try:
        if device.type == 'mps':
            # MPS has some limitations, handle gracefully
            if hasattr(tensor_or_module, 'to'):
                return tensor_or_module.to(device)
            else:
                return tensor_or_module
        else:
            if hasattr(tensor_or_module, 'to'):
                return tensor_or_module.to(device)
            elif hasattr(tensor_or_module, 'cuda') and device.type == 'cuda':
                return tensor_or_module.cuda()
            else:
                return tensor_or_module
    except Exception as e:
        warnings.warn(f"Failed to move to {device}: {e}. Using CPU instead.")
        return tensor_or_module.cpu() if hasattr(tensor_or_module, 'cpu') else tensor_or_module

def get_device_info():
    """
    Get information about available devices
    """
    device = get_device()
    info = {
        'device': device,
        'device_type': get_device_type(),
        'autocast_enabled': is_autocast_enabled(),
        'autocast_dtype': get_autocast_dtype(),
    }
    
    if device.type == 'cuda':
        info.update({
            'device_name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_available': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        })
    elif device.type == 'mps':
        info.update({
            'device_name': 'Apple Silicon GPU (MPS)',
            'memory_total': 'Shared with system',
            'memory_available': 'Shared with system'
        })
    else:
        info.update({
            'device_name': 'CPU',
            'memory_total': 'System RAM',
            'memory_available': 'System RAM'
        })
    
    return info

def print_device_info():
    """
    Print device information
    """
    info = get_device_info()
    print(f"🔧 Device Configuration:")
    print(f"  Device: {info['device']}")
    print(f"  Device Name: {info['device_name']}")
    print(f"  Autocast: {'Enabled' if info['autocast_enabled'] else 'Disabled'}")
    if info['autocast_enabled']:
        print(f"  Autocast dtype: {info['autocast_dtype']}")
    
    if isinstance(info['memory_total'], int):
        print(f"  Memory: {info['memory_total'] / 1e9:.1f}GB total")
    else:
        print(f"  Memory: {info['memory_total']}")

# Global device instance
DEVICE = get_device()
DEVICE_TYPE = get_device_type()
AUTOCAST_ENABLED = is_autocast_enabled()
AUTOCAST_DTYPE = get_autocast_dtype()

# Backwards compatibility
def cuda_if_available():
    """Backwards compatibility function"""
    return get_device()

def to_device(x):
    """Backwards compatibility function"""
    return move_to_device(x) 