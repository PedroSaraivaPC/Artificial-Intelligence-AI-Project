#!/usr/bin/env python3
"""
Move all class images from dataset/validation to dataset/train without collisions.

Behavior:
- Detect class names as subfolders inside dataset/validation/
- For each class, list images in validation/<class>, shuffle, then move into train/<class>
- Renames each moved file to continue the numeric sequence in train/<class>
  Example: if train/<class> has files up to 001532.png, moved files become 001533.png, 001534.png, ...
- Supports extensions: .jpg .jpeg .png .bmp .webp (case-insensitive)
- Prints a summary at the end.

Usage:
    python move_val_to_train.py /path/to/dataset

If you omit the path, it assumes the script is inside the dataset folder and uses the parent directory of 'validation' or 'train' if found.
"""

import sys
import random
from pathlib import Path
import shutil
import re

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_dataset_root(root_hint: Path) -> Path:
    """
    Resolve the dataset root (the folder that contains 'train' and 'validation').
    Priority:
        1) If root_hint has 'train' and/or 'validation', use it directly.
        2) If root_hint contains a subdir 'dataset' that has 'train'/'validation', use that.
        3) If not, but root_hint has a child named 'train' or 'validation', use root_hint.
    """
    # If user passed .../dataset directly
    if (root_hint / 'train').exists() or (root_hint / 'validation').exists():
        return root_hint
    # If inside a parent that has dataset
    if (root_hint / 'dataset').exists() and ((root_hint / 'dataset' / 'train').exists() or (root_hint / 'dataset' / 'validation').exists()):
        return root_hint / 'dataset'
    # Try walking up one level if they passed the script folder inside dataset
    parent = root_hint.parent
    if (parent / 'train').exists() or (parent / 'validation').exists():
        return parent
    raise SystemExit(f"❌ Couldn't find 'train'/'validation' under: {root_hint}. Pass the dataset folder path explicitly.")

def numeric_prefix(stem: str) -> int:
    """
    Extract a leading integer from a filename stem like '00012', '12', '12_something' -> 12
    Return -1 if no leading digits.
    """
    m = re.match(r'^(\d+)', stem)
    return int(m.group(1)) if m else -1

def next_index(train_class_dir: Path) -> int:
    """
    Find the highest leading integer among image filenames in the train class dir and return next value.
    If none found, start from 1.
    """
    max_idx = 0
    for p in train_class_dir.glob('*'):
        if not is_image(p):
            continue
        idx = numeric_prefix(p.stem)
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 1 else 1

def zero_pad(n: int, width: int = 5) -> str:
    return f"{n:0{width}d}"

def collect_classes(validation_dir: Path) -> list[str]:
    classes = [d.name for d in validation_dir.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit(f"❌ No class folders found in {validation_dir}")
    return classes

def main():
    if len(sys.argv) >= 2:
        dataset_root = Path(sys.argv[1]).expanduser().resolve()
    else:
        dataset_root = Path.cwd().resolve()
    dataset_root = find_dataset_root(dataset_root)

    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'validation'

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("❌ Both 'train' and 'validation' directories must exist.")

    classes = collect_classes(val_dir)
    print(f"▶ Classes detected in validation/: {classes}")

    total_moved = 0

    for cls in classes:
        src = val_dir / cls
        dst = train_dir / cls
        if not dst.exists():
            print(f"ℹ️  Creating missing train class folder: {dst}")
            dst.mkdir(parents=True, exist_ok=True)

        val_images = [p for p in src.glob('*') if is_image(p)]
        if not val_images:
            print(f"— No images to move for class '{cls}'. Skipping.")
            continue

        random.shuffle(val_images)

        idx = next_index(dst)
        pad_width = 5  # adjust if you expect > 99,999 images per class

        print(f"• Moving {len(val_images)} images from '{src.name}' to '{dst.name}' starting at index {idx}...")

        moved_for_class = 0
        for p in val_images:
            new_name = f"{zero_pad(idx, pad_width)}{p.suffix.lower()}"
            target = dst / new_name
            # Just in case, bump idx until a free filename is found
            while target.exists():
                idx += 1
                new_name = f"{zero_pad(idx, pad_width)}{p.suffix.lower()}"
                target = dst / new_name
            shutil.move(str(p), str(target))
            idx += 1
            moved_for_class += 1

        total_moved += moved_for_class
        print(f"  ✅ Moved {moved_for_class} images for class '{cls}'.")

        # Optionally remove empty validation class folder
        remaining = [q for q in src.glob('*') if q.is_file()]
        if not remaining:
            try:
                src.rmdir()
                print(f"  🧹 Removed empty folder: {src}")
            except OSError:
                pass

    print("\n================ SUMMARY ================")
    print(f"Total images moved: {total_moved}")
    # If validation/ is empty now, try to remove it
    try:
        leftover = [d for d in val_dir.iterdir()]
        if not leftover:
            val_dir.rmdir()
            print(f"Removed empty 'validation' directory.")
        else:
            # Show what's left
            names = ', '.join([x.name for x in leftover])
            print(f"'validation' still contains: {names}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
