import os
import sys

current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current)

print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

print("Checking 'preprocessing'...")
try:
    import preprocessing
    print("  [OK] import preprocessing")
    if hasattr(preprocessing, '__path__'):
        print(f"  Path: {preprocessing.__path__}")
except ImportError as e:
    print(f"  [FAIL] {e}")

print("Checking 'preprocessing.ecg_preprocessing_pan_tompkins'...")
try:
    import preprocessing.ecg_preprocessing_pan_tompkins
    print("  [OK] import ecg")
except ImportError as e:
    print(f"  [FAIL] {e}")

print("Checking 'training'...")
try:
    import training
    print("  [OK] import training")
except ImportError as e:
    print(f"  [FAIL] {e}")

print("Checking 'training.wesad_dataset'...")
try:
    import training.wesad_dataset
    print("  [OK] import training.wesad_dataset")
except Exception as e:
    print(f"  [FAIL] {e}")

# Check data path
root = os.path.join(current, 'data', 'raw', 'WESAD')
print(f"Checking data root: {root}")
if os.path.exists(root):
    print("  [OK] Root exists")
    # Check content
    try:
        print(f"  Content: {os.listdir(root)}")
    except:
        print("  Error listing root")
else:
    print("  [FAIL] Root does not exist")
