from training.wesad_dataset import WESADDataset
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def verify():
    root = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    print(f"Checking root: {root}")
    if not os.path.exists(root):
        print("Root path does not exist!")
        return
        
    print("Files in root:")
    try:
        print(os.listdir(root))
    except Exception as e:
        print(e)
    
    print("Attempting to load Subject 2...")
    try:
        ds = WESADDataset([2], root)
        print(f"Successfully loaded. Samples: {len(ds)}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
