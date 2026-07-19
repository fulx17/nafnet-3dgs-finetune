"""
Script kiểm tra: đã đặt đúng file, đúng chỗ, đúng nội dung cho việc
fine-tune NAFNet + DiffJPEG + composite loss chưa.
Chạy: python check_setup.py
"""
import os
import sys
import importlib

sys.path.append('.')
sys.path.append('./DiffJPEG')

ROOT = os.path.abspath('.')
ERRORS = []
OK = []


def check_file_exists(path, label):
    full = os.path.join(ROOT, path)
    if os.path.exists(full):
        OK.append(f"[FILE OK] {label}: {path}")
        return True
    else:
        ERRORS.append(f"[MISSING] {label}: {path}")
        return False


def check_import(module_path, class_name, label):
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            ERRORS.append(f"[IMPORT FAIL] {label}: class '{class_name}' not found in {module_path}")
        else:
            OK.append(f"[IMPORT OK] {label}: {module_path}.{class_name}")
        return cls
    except Exception as e:
        ERRORS.append(f"[IMPORT ERROR] {label}: {module_path} -> {type(e).__name__}: {e}")
        return None


print("=" * 60)
print("1. KIEM TRA VI TRI FILE")
print("=" * 60)

check_file_exists("basicsr/data/paired_png16_dataset.py", "Dataset file")
check_file_exists("basicsr/models/nafnet_jpeg_model.py", "Model file")
check_file_exists("basicsr/models/losses/composite_loss.py", "Loss file")
check_file_exists("basicsr/models/losses/__init__.py", "Losses __init__")
check_file_exists("DiffJPEG/DiffJPEG.py", "DiffJPEG module")

print("\n" + "=" * 60)
print("2. KIEM TRA LOSSES __init__.py CO IMPORT EvalAlignedLoss KHONG")
print("=" * 60)

losses_init_path = os.path.join(ROOT, "basicsr/models/losses/__init__.py")
if os.path.exists(losses_init_path):
    with open(losses_init_path, encoding="utf-8") as f:
        content = f.read()
    if "EvalAlignedLoss" in content:
        OK.append("[CONTENT OK] EvalAlignedLoss found in losses/__init__.py")
    else:
        ERRORS.append("[CONTENT MISSING] EvalAlignedLoss NOT imported in losses/__init__.py")

print("\n" + "=" * 60)
print("3. KIEM TRA IMPORT THAT (co load duoc class khong)")
print("=" * 60)

check_import("basicsr.data.paired_png16_dataset", "Paired16BitDataset", "Dataset class")
check_import("basicsr.models.losses", "EvalAlignedLoss", "Loss class (qua losses/__init__.py)")
check_import("basicsr.models.nafnet_jpeg_model", "NAFNetJPEGModel", "Model class")

print("\n" + "=" * 60)
print("4. KIEM TRA basicsr TU DONG SCAN DUOC FILE MOI KHONG")
print("=" * 60)

try:
    from basicsr.data import _dataset_modules
    dataset_names = [m.__name__ for m in _dataset_modules]
    if any("paired_png16_dataset" in n for n in dataset_names):
        OK.append("[SCAN OK] basicsr.data da scan thay paired_png16_dataset")
    else:
        ERRORS.append("[SCAN FAIL] basicsr.data KHONG scan thay paired_png16_dataset "
                       "(kiem tra file co dung duoi '_dataset.py' khong, "
                       "va co nam dung trong basicsr/data/ khong)")
except Exception as e:
    ERRORS.append(f"[SCAN ERROR] Khong the kiem tra basicsr.data scan: {e}")

try:
    from basicsr.models import _model_modules
    model_names = [m.__name__ for m in _model_modules]
    if any("nafnet_jpeg_model" in n for n in model_names):
        OK.append("[SCAN OK] basicsr.models da scan thay nafnet_jpeg_model")
    else:
        ERRORS.append("[SCAN FAIL] basicsr.models KHONG scan thay nafnet_jpeg_model "
                       "(kiem tra file co dung duoi '_model.py' khong, "
                       "va co nam dung trong basicsr/models/ khong)")
except Exception as e:
    ERRORS.append(f"[SCAN ERROR] Khong the kiem tra basicsr.models scan: {e}")

print("\n" + "=" * 60)
print("KET QUA")
print("=" * 60)

for line in OK:
    print(line)

print()
if ERRORS:
    for line in ERRORS:
        print(line)
    print(f"\n>>> CO {len(ERRORS)} LOI CAN SUA <<<")
else:
    print(">>> TAT CA OK, SAN SANG BUOC TIEP THEO <<<")
    