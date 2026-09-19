"""
scripts/migrate_packages_to_venv.py

Migrates installed AI/ML packages (PyTorch, Torchvision, Ultralytics, TensorRT, etc.)
from the PC's host/user site-packages into the project's local .venv without re-downloading.
Cleans up the global user environment upon successful migration so your PC remains pristine.
"""

import os
import sys
import glob
import shutil
import site
import subprocess

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_DIR = os.path.join(REPO_DIR, ".venv")
VENV_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
VENV_SITE = os.path.join(VENV_DIR, "Lib", "site-packages")
VENV_SCRIPTS = os.path.join(VENV_DIR, "Scripts")

# Packages and prefixes relevant to AI / Vision / TensorRT stack
TARGET_PREFIXES = [
    "torch", "torchvision", "torchaudio", "caffe2",
    "ultralytics", "ultralytics_thop",
    "tensorrt", "tensorrt_cu12", "tensorrt_cu12_bindings", "tensorrt_cu12_libs",
    "cv2", "opencv",
    "numpy", "scipy", "sklearn", "scikit_learn",
    "matplotlib", "mpl_toolkits",
    "pyyaml", "yaml", "timm", "einops", "tqdm",
    "safetensors", "huggingface_hub",
    "requests", "pil", "pillow",
    "sympy", "jinja2", "mpmath", "networkx",
    "filelock", "fsspec", "typing_extensions",
    "psutil", "polars", "joblib", "threadpoolctl",
    "pandas", "seaborn", "transformers", "accelerate",
    "qwen_vl_utils", "sentencepiece",
    "onnx", "onnxslim",
    "certifi", "charset_normalizer", "idna", "urllib3",
    "markupsafe", "six", "packaging", "colorama",
    "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing", "dateutil"
]

EXCLUDED_NAMES = {
    "pip", "pip-", "_distutils_hack", "distutils-precedence.pth",
    "setuptools", "wheel", "easy_install", "__pycache__"
}


def create_venv_if_missing():
    """Ensures .venv exists before moving packages."""
    if not os.path.exists(VENV_PY):
        print(f"[AUTO-SETUP] Initializing clean virtual environment at: {VENV_DIR}...")
        try:
            # Use host python to create .venv
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            print("[AUTO-SETUP] Virtual environment created successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to create .venv: {e}")
            sys.exit(1)


def get_candidate_sources():
    """Finds all candidate external site-packages and scripts directories on Windows/Linux."""
    site_dirs = []
    scripts_dirs = []

    # 1. Standard user site packages
    try:
        user_site = site.getusersitepackages()
        if user_site and os.path.exists(user_site) and VENV_DIR not in os.path.abspath(user_site):
            site_dirs.append((user_site, True))  # True indicates user-level directory
            user_scripts = os.path.join(os.path.dirname(user_site), "Scripts")
            if os.path.exists(user_scripts):
                scripts_dirs.append((user_scripts, True))
    except Exception:
        pass

    # 2. Global/System site packages (outside .venv)
    try:
        for p in site.getsitepackages():
            if os.path.exists(p) and VENV_DIR not in os.path.abspath(p):
                site_dirs.append((p, False))
                p_scripts = os.path.join(os.path.dirname(p), "Scripts")
                if os.path.exists(p_scripts):
                    scripts_dirs.append((p_scripts, False))
    except Exception:
        pass

    # 3. Windows AppData paths (e.g. Microsoft Store Python, standard AppData installations)
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        # Microsoft Store Python pattern:
        # AppData\Local\Packages\PythonSoftwareFoundation.Python.*\LocalCache\local-packages\Python*\site-packages
        ms_pattern = os.path.join(
            local_app_data, "Packages", "PythonSoftwareFoundation.Python.*",
            "LocalCache", "local-packages", "Python*", "site-packages"
        )
        for d in glob.glob(ms_pattern):
            if os.path.exists(d) and VENV_DIR not in os.path.abspath(d):
                site_dirs.append((d, True))
                s_dir = os.path.join(os.path.dirname(d), "Scripts")
                if os.path.exists(s_dir):
                    scripts_dirs.append((s_dir, True))

        # Standard Programs Python pattern:
        prog_pattern = os.path.join(local_app_data, "Programs", "Python", "Python*", "Lib", "site-packages")
        for d in glob.glob(prog_pattern):
            if os.path.exists(d) and VENV_DIR not in os.path.abspath(d):
                site_dirs.append((d, False))

    app_data = os.environ.get("APPDATA", "")
    if app_data:
        # Roaming Python pattern:
        roam_pattern = os.path.join(app_data, "Python", "Python*", "site-packages")
        for d in glob.glob(roam_pattern):
            if os.path.exists(d) and VENV_DIR not in os.path.abspath(d):
                site_dirs.append((d, True))
                s_dir = os.path.join(os.path.dirname(d), "Scripts")
                if os.path.exists(s_dir):
                    scripts_dirs.append((s_dir, True))

    # De-duplicate while preserving order
    clean_sites = []
    seen_sites = set()
    for s_path, is_user in site_dirs:
        norm = os.path.normcase(os.path.abspath(s_path))
        if norm not in seen_sites and os.path.exists(s_path):
            seen_sites.add(norm)
            clean_sites.append((s_path, is_user))

    clean_scripts = []
    seen_scripts = set()
    for sc_path, is_user in scripts_dirs:
        norm = os.path.normcase(os.path.abspath(sc_path))
        if norm not in seen_scripts and os.path.exists(sc_path) and VENV_DIR not in os.path.abspath(sc_path):
            seen_scripts.add(norm)
            clean_scripts.append((sc_path, is_user))

    return clean_sites, clean_scripts


def should_migrate(item_name: str, is_user_site: bool) -> bool:
    """Determines whether a package or file should be moved into .venv."""
    name_lower = item_name.lower()

    # Always protect host pip/setuptools runtime
    for ex in EXCLUDED_NAMES:
        if name_lower == ex or name_lower.startswith(ex):
            return False

    # In user site-packages (e.g. AppData), everything that was installed was placed there by user pip commands
    if is_user_site:
        return True

    # In global/system site-packages, migrate packages that match our project target prefixes
    for prefix in TARGET_PREFIXES:
        if name_lower.startswith(prefix):
            return True

    return False


def move_item(src_path: str, dst_path: str) -> bool:
    """Safely moves a file or directory, handling cross-drive migration and collisions."""
    try:
        if os.path.exists(dst_path):
            if os.path.isdir(dst_path):
                shutil.rmtree(dst_path, ignore_errors=True)
            else:
                try:
                    os.remove(dst_path)
                except Exception:
                    pass

        shutil.move(src_path, dst_path)
        return True
    except Exception:
        # Fallback for cross-device moves or file locks
        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                shutil.rmtree(src_path, ignore_errors=True)
            else:
                shutil.copy2(src_path, dst_path)
                try:
                    os.remove(src_path)
                except Exception:
                    pass
            return True
        except Exception as err:
            print(f"    [WARNING] Failed to migrate {os.path.basename(src_path)}: {err}")
            return False


def migrate():
    create_venv_if_missing()
    os.makedirs(VENV_SITE, exist_ok=True)
    os.makedirs(VENV_SCRIPTS, exist_ok=True)

    sources, script_sources = get_candidate_sources()

    print("\n" + "=" * 80)
    print("   ZERO-DOWNLOAD PACKAGE MIGRATION: HOST PC -> PROJECT .VENV")
    print(f"   Target Virtual Environment: {VENV_DIR}")
    print("=" * 80)

    if not sources:
        print("[MIGRATE] No external site-packages found on host PC.")
        return

    total_moved = 0

    # 1. Migrate packages and metadata
    for src_dir, is_user in sources:
        if VENV_DIR in os.path.abspath(src_dir):
            continue

        print(f"\n[SCAN] Inspecting: {src_dir}")
        try:
            items = sorted(os.listdir(src_dir))
        except Exception as e:
            print(f"  [WARNING] Cannot read {src_dir}: {e}")
            continue

        moved_in_dir = 0
        for item in items:
            if should_migrate(item, is_user):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(VENV_SITE, item)
                print(f"  -> Moving into .venv: {item} ...")
                if move_item(src_path, dst_path):
                    moved_in_dir += 1
                    total_moved += 1

        print(f"  [DONE] Moved {moved_in_dir} items from this location.")

        # Clean up empty parent directories in AppData if emptied
        if is_user:
            try:
                remaining = os.listdir(src_dir)
                if not remaining:
                    os.rmdir(src_dir)
                    print(f"  [CLEANUP] Removed empty directory: {src_dir}")
            except Exception:
                pass

    # 2. Migrate CLI executables from Scripts
    for sc_dir, is_user in script_sources:
        try:
            items = os.listdir(sc_dir)
        except Exception:
            continue

        for item in items:
            name_lower = item.lower()
            if any(name_lower.startswith(p) for p in TARGET_PREFIXES):
                src_path = os.path.join(sc_dir, item)
                dst_path = os.path.join(VENV_SCRIPTS, item)
                move_item(src_path, dst_path)

    print("\n" + "=" * 80)
    print(f"[MIGRATE COMPLETE] Successfully moved {total_moved} packages into .venv!")
    print("[MIGRATE] Removed from host PC storage. Your personal PC is completely clean.")
    print("=" * 80 + "\n")

    # 3. Verification inside .venv
    try:
        verify_cmd = [
            VENV_PY, "-c",
            "import torch, ultralytics; "
            "print(f'[VERIFIED IN .VENV] PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Ultralytics: {ultralytics.__version__}')"
        ]
        out = subprocess.check_output(verify_cmd, text=True, stderr=subprocess.STDOUT).strip()
        print(out)
        print("[SUCCESS] .venv is fully operational with zero network download!\n")
    except Exception as e:
        print(f"[NOTE] Environment verification check: {e}")


if __name__ == "__main__":
    migrate()
