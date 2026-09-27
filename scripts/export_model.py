#!/usr/bin/env python
"""
export_model.py - Standalone Model Exporter for Edge Deployment
Exports models/yolo8m.pt to ONNX format while validating SH17 17-class metadata.
"""

import os
import sys
from ultralytics import YOLO

VERIFIED_MODEL_CLASSES = {
    0: 'person',    1: 'ear',          2: 'ear-mufs',   3: 'face',
    4: 'face-guard',5: 'face-mask',    6: 'foot',       7: 'tool',
    8: 'glasses',   9: 'gloves',      10: 'helmet',    11: 'hands',
   12: 'head',     13: 'medical-suit',14: 'shoes',     15: 'safety-suit',
   16: 'safety-vest'
}

def export_model(model_path="models/yolo8m.pt", export_format="onnx", opset=17, imgsz=640):
    if not os.path.exists(model_path):
        print(f"FATAL: Model weights not found at: {model_path}")
        sys.exit(1)

    print(f"Loading SH17 model from: {model_path}")
    model = YOLO(model_path)

    # Validate class schema
    mismatches = [
        (k, VERIFIED_MODEL_CLASSES[k], model.names.get(k))
        for k in VERIFIED_MODEL_CLASSES
        if model.names.get(k) != VERIFIED_MODEL_CLASSES[k]
    ]
    if mismatches:
        print("FATAL: Class schema validation failed. Export aborted.")
        for k, exp, got in mismatches:
            print(f"  class {k}: expected '{exp}', got '{got}'")
        sys.exit(1)

    print(f"Model schema validated: {len(model.names)} SH17 classes confirmed.")
    print(f"Exporting to format={export_format} (opset={opset}, imgsz={imgsz})...")

    exported_path = model.export(
        format=export_format,
        opset=opset,
        imgsz=imgsz,
        simplify=True
    )
    print(f"Export successful! Exported model saved to: {exported_path}")
    return exported_path

if __name__ == "__main__":
    format_choice = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    export_model(export_format=format_choice)
