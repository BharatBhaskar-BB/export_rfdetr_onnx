#!/usr/bin/env python3
"""
export_onnx.py — RF-DETR -> ONNX exporter (roboflow/rf-detr compatible)

Features:
- Uses rfdetr.main.get_args_parser() defaults
- Auto-injects missing args required by rfdetr.models.build_model(args)
  (keeps retrying when it sees AttributeError: Namespace has no attribute 'XYZ')
- Loads .pth weights best-effort (supports common checkpoint formats)
- Exports ONNX with dynamic axes (batch/height/width)
- Prints ONNXRuntime input/output tensor names + shapes after export

Usage:
  source .venv/bin/activate

  python export_onnx.py \
    --weights /ABS/PATH/rf-detr-nano.pth \
    --out backend/app/models/rfdetr-nano.onnx \
    --imgsz 640

If you want to just inspect repo:
  python export_onnx.py --introspect
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch


# ------------------------- helpers -------------------------

def _banner(msg: str) -> None:
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90 + "\n")


def introspect_rfdetr() -> None:
    _banner("RF-DETR introspection")
    import pkgutil

    try:
        import rfdetr  # type: ignore
        print("rfdetr imported from:", rfdetr.__file__)
    except Exception as e:
        print("FAILED to import rfdetr:", repr(e))
        print("Install it into the active venv, e.g.:")
        print("  python -m pip install -e ./rf-detr")
        sys.exit(1)

    print("\nTop-level submodules under 'rfdetr':")
    for m in pkgutil.iter_modules(rfdetr.__path__):  # type: ignore
        print("  -", m.name)

    try:
        from rfdetr.models import build_model  # type: ignore
        import inspect
        print("\nbuild_model signature:", inspect.signature(build_model))
    except Exception as e:
        print("\nCould not import build_model:", repr(e))


def load_checkpoint_state_dict(weights_path: str) -> Dict[str, Any]:
    """Loads a checkpoint and returns a state_dict-like mapping (safe by default)."""
    import argparse as _argparse
    import torch

    # PyTorch 2.6+ defaults to weights_only=True; we keep it and allowlist Namespace.
    try:
        torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass

    # 1) Safe load first (no arbitrary code exec)
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions might not support weights_only
        ckpt = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        # 2) If safe load fails, try unsafe load ONLY if file is trusted
        print("[warn] Safe weights-only load failed:", repr(e))
        print("[warn] Trying torch.load(..., weights_only=False). Do this ONLY for trusted checkpoints.")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Now extract state_dict
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

        # Some trainers save the state dict under other common keys
        for k in ("model_state", "model_state_dict", "net", "network"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]

        # If it already looks like a state_dict, return it
        # (heuristic: keys look like parameter names)
        if any(isinstance(k, str) and "." in k for k in ckpt.keys()):
            return ckpt

    raise RuntimeError(f"Unsupported checkpoint format or no state_dict found. Type={type(ckpt)} keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'n/a'}")



def default_for_missing_key(key: str, args: argparse.Namespace) -> Any:
    """
    Provide a sensible default for missing args.<key>.
    These defaults are conservative and chosen to get the model to build.
    """
    # Exact known keys we already saw
    if key == "gradient_checkpointing":
        return False
    if key == "patch_size":
        return 16
    if key == "num_windows":
        return 0
    if key == "window_size":
        return 0
    if key == "positional_encoding_size":
        # safe default: match hidden_dim if available, else 256
        return getattr(args, "hidden_dim", 256)
    if key == "load_dinov2_weights":
        return False
    if key == "block_size":
        return 16

    # Heuristics by name patterns
    lk = key.lower()

    # booleans
    if lk.startswith(("use_", "enable_", "do_", "with_", "has_")):
        return False

    # dropout / rates
    if any(tok in lk for tok in ("drop", "dropout", "drop_path", "rate")):
        return 0.0

    # sizes / dims
    if any(tok in lk for tok in ("dim", "width", "height", "size", "embed", "hidden")):
        # hidden_dim is a good safe base in this repo
        return getattr(args, "hidden_dim", 256)

    # counts
    if lk.startswith("num_") or lk.endswith(("_layers", "_heads", "_points", "_queries")):
        # try to infer from related args or use a safe small number
        if "layers" in lk:
            return 1
        if "heads" in lk:
            return 1
        return 0

    # schedules / modes / strings
    if lk.endswith(("mode", "type", "name", "schedule", "norm")):
        return None

    # fallback
    return None


# --------------------- model wrapper -----------------------

class ExportWrapper(torch.nn.Module):
    """
    Wrap model so ONNX export returns tensors/tuples.

    RF-DETR/DETR models often return dicts like:
      {"pred_logits": [B,Q,C], "pred_boxes": [B,Q,4], ...}
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        out = self.model(images)

        # DETR-style dict
        if isinstance(out, dict):
            if "pred_boxes" in out and "pred_logits" in out:
                return out["pred_boxes"], out["pred_logits"]

            # Some variants might output already-decoded fields
            if "boxes" in out and "scores" in out:
                boxes = out["boxes"]
                scores = out["scores"]
                labels = out.get("labels", torch.zeros((scores.shape[0],), dtype=torch.int64, device=scores.device))
                return boxes, scores, labels

            # Fallback: return first two tensors found
            tensors = [v for v in out.values() if torch.is_tensor(v)]
            if len(tensors) >= 2:
                return tensors[0], tensors[1]
            if len(tensors) == 1:
                return tensors[0]
            raise RuntimeError(f"Output dict has no tensors. Keys={list(out.keys())}")

        # Already tuple/list of tensors
        if isinstance(out, (tuple, list)):
            return tuple(out)

        # Single tensor
        if torch.is_tensor(out):
            return out

        raise RuntimeError(f"Unsupported model output type: {type(out)}")


# --------------------- builder + export --------------------

def build_rfdetr_model_autofill(verbose: bool = True) -> torch.nn.Module:
    """
    Build RF-DETR model by using get_args_parser() defaults, forcing CPU,
    and auto-filling any missing args accessed by build_model(args).
    """
    from rfdetr.models import build_model  # type: ignore

    # Load default args from repo parser
    args: argparse.Namespace
    try:
        from rfdetr.main import get_args_parser  # type: ignore
        parser = get_args_parser()
        args = parser.parse_args([])
    except Exception:
        args = argparse.Namespace()

    # Force CPU for export stability
    args.device = "cpu"
    # ---- Prevent modulo-by-zero in dinov2 backbone ----
    # Some configs compute block_size from window settings; ensure it's non-zero.
    if not hasattr(args, "block_size") or getattr(args, "block_size") in (None, 0):
        args.block_size = getattr(args, "patch_size", 16)  # safe default

    # Also ensure window parameters don't imply block_size=0
    if not hasattr(args, "num_windows") or args.num_windows is None:
        args.num_windows = 0
    if not hasattr(args, "window_size") or args.window_size is None:
        args.window_size = 0

    # --- Make model match your checkpoint (COCO + patch16 + resolution384) ---
    args.num_classes = 90          # => class_embed becomes 91
    args.patch_size = 16           # checkpoint has 16x16 patch projection
    args.resolution = 384          # checkpoint pos_embed length 577 implies 384 with patch16

    # --- Make backbone features match projector input (1536 = 384*4 stages) ---
    args.out_feature_indexes = [3, 6, 9, 12]

    # --- Ensure projector_scale is list form ---
    if isinstance(getattr(args, "projector_scale", "P4"), str):
        args.projector_scale = [args.projector_scale]




    # ---- Fix projector_scale: must be a list like ["P4"], not the string "P4" ----
    if hasattr(args, "projector_scale"):
        if isinstance(args.projector_scale, str):
            args.projector_scale = [args.projector_scale]
    else:
        args.projector_scale = ["P4"]

    # Ensure valid + ascending + only supported levels
    allowed = {"P3", "P4", "P5", "P6"}
    args.projector_scale = [p for p in args.projector_scale if p in allowed]
    if not args.projector_scale:
        args.projector_scale = ["P4"]
    args.projector_scale = sorted(args.projector_scale, key=lambda x: int(x[1]))



    # ---- Fix invalid out_feature_indexes for HF AutoBackbone ----
    # Repo default uses [-1], but Transformers expects stage names like stage1..stage12.
    # Convert "-1" (last) into a valid stage index (12).
    if not hasattr(args, "out_feature_indexes") or args.out_feature_indexes is None:
        args.out_feature_indexes = [12]
    else:
        # Replace -1 with last stage
        args.out_feature_indexes = [12 if int(i) == -1 else int(i) for i in args.out_feature_indexes]

    # For single-scale export, keep only one stage (more stable)
    if len(args.out_feature_indexes) > 1:
        args.out_feature_indexes = [max(args.out_feature_indexes)]
    # --- IMPORTANT: this repo expects DINOv2-style encoder names ---
    # Use a lightweight backbone for "nano" style checkpoints.
    args.encoder = "dinov2_small"

    # If the backbone code tries to download DINOv2 weights from HF hub,
    # disable it because we are loading local RF-DETR checkpoint weights.
    # (This arg may not exist in parser; we set it explicitly.)
    args.load_dinov2_weights = False


    # Ensure some sane baseline keys
    baseline_defaults = {
        "gradient_checkpointing": False,
        "patch_size": 16,
        "num_windows": 0,
        "window_size": 0,
        "positional_encoding_size": getattr(args, "hidden_dim", 256),
    }
    for k, v in baseline_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Now attempt to build; if missing arg, auto-inject and retry
    max_missing = 50
    for attempt in range(max_missing):
        try:
            out = build_model(args)
            model = out[0] if isinstance(out, (tuple, list)) else out
            if verbose:
                print(f"[build_model] success after {attempt} auto-default additions")
            return model
        except AttributeError as e:
            m = re.search(r"has no attribute '([^']+)'", str(e))
            if not m:
                raise
            key = m.group(1)
            if hasattr(args, key):
                # Something else is wrong; re-raise
                raise
            val = default_for_missing_key(key, args)
            setattr(args, key, val)
            if verbose:
                print(f"[auto-default] added args.{key} = {val!r}")
            continue

    raise RuntimeError(f"Exceeded {max_missing} auto-default additions while building model.")


def export_onnx(weights: str, out_path: str, imgsz: int, opset: int, dynamic: bool) -> None:
    _banner("Building RF-DETR model")
    model = build_rfdetr_model_autofill(verbose=True)
    model.eval()

    if weights:
        _banner(f"Loading weights: {weights}")
        sd = load_checkpoint_state_dict(weights)
        # missing, unexpected = model.load_state_dict(sd, strict=False)
        # ---- Robust partial weight loading (handles backbone / projector mismatches) ----
        model_sd = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        print(f"[weights] loaded {len(filtered)} tensors, skipped {len(skipped)} incompatible tensors")

        model.load_state_dict(filtered, strict=False)



        # print("Loaded state_dict with strict=False")
        # print("Missing keys:", len(missing))
        # print("Unexpected keys:", len(unexpected))
    else:
        print("No weights provided; exporting random-initialized model (not useful for detection).")

    wrapper = ExportWrapper(model).eval()

    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)

    out_path = str(Path(out_path).resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    _banner(f"Exporting ONNX -> {out_path}")

    # Dry run to know number of outputs
    with torch.no_grad():
        y = wrapper(dummy)

    if torch.is_tensor(y):
        num_out = 1
    elif isinstance(y, (tuple, list)):
        num_out = len(y)
    else:
        raise RuntimeError("Unexpected wrapper output type")

    input_names = ["images"]
    output_names = [f"out{i}" for i in range(num_out)]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    _banner("ONNX export done. Checking with ONNXRuntime (names + shapes)")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        print("Inputs:")
        for i in sess.get_inputs():
            print(" ", i.name, i.shape, i.type)
        print("Outputs:")
        for o in sess.get_outputs():
            print(" ", o.name, o.shape, o.type)
    except Exception as e:
        print("ONNXRuntime check failed:", repr(e))
        print("But the ONNX file may still be created. Verify the file exists.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="", help="Path to RF-DETR .pth checkpoint")
    ap.add_argument("--out", type=str, default="rfdetr.onnx", help="Output ONNX path")
    ap.add_argument("--imgsz", type=int, default=640, help="Square inference size for export")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--no-dynamic", action="store_true", help="Disable dynamic axes")
    ap.add_argument("--introspect", action="store_true", help="Print rfdetr package info and exit")
    args = ap.parse_args()

    if args.introspect:
        introspect_rfdetr()
        return

    weights = args.weights.strip()
    if weights and not os.path.exists(weights):
        raise SystemExit(f"Weights file not found: {weights}")

    export_onnx(
        weights=weights,
        out_path=args.out,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=not args.no_dynamic,
    )


if __name__ == "__main__":
    main()
