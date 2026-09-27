#!/usr/bin/env python
"""scripts/validate_model.py — Standalone SH17 model validator.

Run this to confirm that models/yolo8m.pt is the correct SH17 17-class model
before any inference. Exits with code 0 on success, code 1 on any failure.

Usage:
    python scripts/validate_model.py
    python scripts/validate_model.py --model models/yolo8m.pt   # override path

This script can also be imported to get validate_model() as a callable:
    from scripts.validate_model import validate_model, SH17_CLASSES
"""

import os
import sys

# --- Verified SH17 class mapping (source of truth: 2026-08-25) ---
# These are the exact model.names from models/yolo8m.pt.
# HARD RULE: never modify this dict to match a different model.
# If you need a different model, update this dict AND re-verify with model.names.
SH17_CLASSES = {
    0:  'person',       1:  'ear',          2:  'ear-mufs',
    3:  'face',         4:  'face-guard',   5:  'face-mask',
    6:  'foot',         7:  'tool',         8:  'glasses',
    9:  'gloves',       10: 'helmet',       11: 'hands',
    12: 'head',         13: 'medical-suit', 14: 'shoes',
    15: 'safety-suit',  16: 'safety-vest',
}

# All SH17 classes that are PPE equipment (anatomy + tool excluded)
SH17_PPE_CLASSES = {
    'ear-mufs', 'face-guard', 'face-mask', 'glasses', 'gloves',
    'helmet', 'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
}

# Prototype compliance policy baseline
# Change REQUIRED_PPE to change the active policy — do not touch SH17_CLASSES or the algorithm.
REQUIRED_PPE = {'helmet', 'safety-vest', 'gloves', 'shoes', 'face-mask'}


def validate_model(model, expected_classes=None):
    """Validate a loaded YOLO model has the expected SH17 class schema.

    Args:
        model: loaded ultralytics YOLO instance (model.names must be populated)
        expected_classes: dict {int: str} to compare against; defaults to SH17_CLASSES

    Returns:
        True if validation passes.

    Raises:
        SystemExit(1) with a detailed message if validation fails.
        Never silently falls back to auto-downloading COCO weights.
    """
    if expected_classes is None:
        expected_classes = SH17_CLASSES

    actual = model.names
    mismatches = [
        (k, expected_classes[k], actual.get(k))
        for k in expected_classes
        if actual.get(k) != expected_classes[k]
    ]
    missing = [k for k in expected_classes if k not in actual]

    if mismatches or missing:
        print("FATAL: Model class validation FAILED.")
        print("  This is NOT the SH17 model. Do NOT substitute COCO yolov8m.pt weights.")
        print("  Obtain the correct SH17 weights (models/yolo8m.pt).")
        if mismatches:
            print("  Class name mismatches:")
            for k, exp, got in mismatches:
                print(f"    class {k}: expected '{exp}', got '{got}'")
        if missing:
            print("  Missing class IDs:")
            for k in missing:
                print(f"    class {k}: expected '{expected_classes[k]}', MISSING from model.names")
        raise SystemExit(1)

    print(f"Model validation PASSED — {len(actual)} SH17 classes confirmed.")
    return True


def main():
    import argparse
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Validate SH17 YOLO model class schema.")
    parser.add_argument("--model", default=os.path.join(os.getcwd(), "models", "yolo8m.pt"),
                        help="Path to the model file (default: models/yolo8m.pt)")
    args = parser.parse_args()

    model_path = args.model
    print(f"Checking model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"FATAL: Model file not found: {model_path}")
        print("  Do NOT download a replacement. Locate the original SH17 weights.")
        sys.exit(1)
    print(f"  Found: {os.path.getsize(model_path):,} bytes")

    print("Loading model...")
    model = YOLO(model_path)
    validate_model(model)

    print("\nSH17 class mapping:")
    for k, v in sorted(model.names.items()):
        ppe_tag = " [PPE]" if v in SH17_PPE_CLASSES else ""
        req_tag = " [REQUIRED]" if v in REQUIRED_PPE else ""
        print(f"  {k:2d}: {v}{ppe_tag}{req_tag}")

    print("\nActive compliance policy (REQUIRED_PPE):", sorted(REQUIRED_PPE))
    print("\nValidation complete. Exit code 0.")


if __name__ == "__main__":
    main()
