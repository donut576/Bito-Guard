# -*- coding: utf-8 -*-
"""
一次執行三個模型腳本：
- model_lightgbm.py
- model_xgboost.py
- model_random_forest.py

用法：
  python run_all_models.py

可選：
  你也可以在 MODELS 裡調整順序、或改成你實際的檔名。
"""

import sys
import subprocess
from pathlib import Path


MODELS = [
    "model_lightgbm.py",
    "model_xgboost.py",
    "model_random_forest.py",
]


def run_script(script_path: Path) -> int:
    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    if not script_path.exists():
        print(f"[ERROR] File not found: {script_path}")
        return 2

    # 用同一個 python 直譯器跑（避免環境不一致）
    cmd = [sys.executable, str(script_path)]

    # 直接把子程序輸出串到目前 console（你就會看到跟單獨跑一樣的輸出）
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    root = Path(__file__).resolve().parent

    failed = []
    for name in MODELS:
        code = run_script(root / name)
        if code != 0:
            failed.append((name, code))

    print("\n" + "-" * 80)
    if failed:
        print("[DONE] Some scripts failed:")
        for name, code in failed:
            print(f"  - {name}: exit code {code}")
        sys.exit(1)
    else:
        print("[DONE] All scripts finished successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()