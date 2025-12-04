import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    print("✅ PyTorch安装成功")
except Exception as e:
    print(f"❌ PyTorch安装失败: {e}")

try:
    import numpy as np
    print(f"NumPy版本: {np.__version__}")
    print("✅ NumPy安装成功")
except Exception as e:
    print(f"❌ NumPy安装失败: {e}")
