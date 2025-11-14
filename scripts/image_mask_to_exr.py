# python
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

def srgb_to_linear_uint8_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    c = rgb_uint8.astype(np.float32) / 255.0
    mask = c <= 0.04045
    c_lin = np.where(mask, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c_lin

def to_float_mask(mask_uint8: np.ndarray) -> np.ndarray:
    if mask_uint8.ndim == 3:
        # If mask accidentally RGB, convert to single channel by luminance
        mask_uint8 = (0.299 * mask_uint8[..., 0] +
                      0.587 * mask_uint8[..., 1] +
                      0.114 * mask_uint8[..., 2]).astype(np.uint8)
    return (mask_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)

def write_exr_openexr(out_path: Path, rgba: np.ndarray):
    try:
        import OpenEXR, Imath  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenEXR not available") from e

    h, w, _ = rgba.shape
    header = OpenEXR.Header(w, h)
    pt_half = Imath.PixelType(Imath.PixelType.HALF)
    header['channels'] = {
        'R': Imath.Channel(pt_half),
        'G': Imath.Channel(pt_half),
        'B': Imath.Channel(pt_half),
        'A': Imath.Channel(pt_half),
    }

    # Convert to float16 per channel and write
    R = np.ascontiguousarray(rgba[..., 0].astype(np.float16)).tobytes()
    G = np.ascontiguousarray(rgba[..., 1].astype(np.float16)).tobytes()
    B = np.ascontiguousarray(rgba[..., 2].astype(np.float16)).tobytes()
    A = np.ascontiguousarray(rgba[..., 3].astype(np.float16)).tobytes()

    exr = OpenEXR.OutputFile(str(out_path), header)
    exr.writePixels({'R': R, 'G': G, 'B': B, 'A': A})
    exr.close()

def write_exr_pyexr(out_path: Path, rgba: np.ndarray):
    try:
        import pyexr  # type: ignore
    except Exception as e:
        raise RuntimeError("pyexr not available") from e
    # pyexr writes FLOAT by default and preserves RGBA
    pyexr.write(str(out_path), rgba.astype(np.float32))

def find_mask_for(stem: str, mask_dir: Path, mask_exts=('png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG')) -> Path | None:
    for ext in mask_exts:
        p = mask_dir / f"{stem}.{ext}"
        if p.exists():
            return p
    return None

def process_folder(input_dir: Path, output_dir: Path, images_subdir: str, masks_subdir: str, linearize: bool, overwrite: bool):
    img_dir = input_dir / images_subdir
    msk_dir = input_dir / masks_subdir

    if not img_dir.is_dir():
        print(f"Missing images folder `{img_dir}`", file=sys.stderr)
        sys.exit(1)
    if not msk_dir.is_dir():
        print(f"Missing masks folder `{msk_dir}`", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.exr')])
    if not image_files:
        print(f"No supported images (JPG/PNG/EXR) found in `{img_dir}`", file=sys.stderr)

        # print(f"No JPG images found in `{img_dir}`", file=sys.stderr)
        return

    # Select writer: OpenEXR preferred, else pyexr
    writer = None
    try:
        import OpenEXR  # noqa: F401
        writer = write_exr_openexr
    except Exception:
        try:
            import pyexr  # noqa: F401
            writer = write_exr_pyexr
            print("OpenEXR not found, using pyexr.", file=sys.stderr)
        except Exception:
            print("Neither OpenEXR nor pyexr is available. Please `pip install openexr Imath` or `pip install pyexr`.", file=sys.stderr)
            sys.exit(1)

    for img_path in image_files:
        stem = img_path.stem
        mask_path = find_mask_for(stem, msk_dir)
        if mask_path is None:
            print(f"Mask not found for `{stem}`; skipping.", file=sys.stderr)
            continue

        out_path = output_dir / f"{stem}.exr"
        if out_path.exists() and not overwrite:
            print(f"Exists `{out_path}`, skipping. Use --overwrite to replace.", file=sys.stderr)
            continue

        # Load image (handles EXR or raster) and mask
        try:
            rgb_f = load_rgb_image(img_path, linearize=linearize)
        except Exception as e:
            print(f"Failed to load image `{img_path}`: {e}", file=sys.stderr)
            continue

        with Image.open(mask_path) as mm:
            mm = mm.convert('L')
            if mm.size != (rgb_f.shape[1], rgb_f.shape[0]):
                mm = mm.resize((rgb_f.shape[1], rgb_f.shape[0]), resample=Image.NEAREST)
            mask_u8 = np.array(mm, dtype=np.uint8)  # HxW

        # # Convert to float
        # if linearize:
        #     rgb_f = srgb_to_linear_uint8_rgb(rgb)  # HxWx3 in [0,1]
        # else:
        #     rgb_f = (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
        #     # rgb_f = rgb

        alpha = to_float_mask(mask_u8)  # HxW in [0,1]
        rgba = np.dstack([rgb_f, alpha])  # HxWx4

        # Write EXR
        writer(out_path, rgba)
        print(f"Wrote `{out_path}`")

def read_exr_rgb(path: Path) -> np.ndarray:
    """Read RGB from an EXR file into float32 array HxWx3 clipped to [0,1]."""
    # Try pyexr first (simpler)
    try:
        import pyexr  # type: ignore
        data = pyexr.open(str(path)).read()  # HxWxC
        if data.ndim != 3 or data.shape[2] < 3:
            raise RuntimeError("EXR missing RGB channels")
        rgb = data[..., :3].astype(np.float32)
        return np.clip(rgb, 0.0, 1.0)
    except Exception:
        pass
    # Fallback: OpenEXR
    try:
        import OpenEXR, Imath  # type: ignore
        exr = OpenEXR.InputFile(str(path))
        dw = exr.header()['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        header_channels = exr.header()['channels'].keys()
        def ch(name: str) -> np.ndarray:
            if name in header_channels:
                buf = exr.channel(name, pt)
                arr = np.frombuffer(buf, dtype=np.float32).reshape(h, w)
            else:
                arr = np.zeros((h, w), dtype=np.float32)
            return arr
        R = ch('R'); G = ch('G'); B = ch('B')
        rgb = np.stack([R, G, B], axis=-1)
        exr.close()
        return np.clip(rgb, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Failed to read EXR `{path}`: {e}") from e

def load_rgb_image(path: Path, linearize: bool) -> np.ndarray:
    """Load image (JPG/PNG/EXR) and return linear RGB float32 HxWx3 in [0,1]."""
    suf = path.suffix.lower()
    if suf == '.exr':
        # EXR assumed already linear
        return read_exr_rgb(path)
    # Raster (sRGB)
    with Image.open(path) as im:
        im = im.convert('RGB')
        rgb_u8 = np.array(im, dtype=np.uint8)
    if linearize:
        return srgb_to_linear_uint8_rgb(rgb_u8)
    return (rgb_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Convert JPG + mask to RGBA EXR.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input root folder containing `images/` and `masks/`.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output folder for .exr files.")
    parser.add_argument("--images-subdir", default="images", help="Subfolder under input for JPGs (default: images).")
    parser.add_argument("--masks-subdir", default="masks", help="Subfolder under input for masks (default: masks).")
    parser.add_argument("--linearize-srgb", action="store_true",
                        help="Convert JPG/PNG sRGB to linear (ignored for EXR input).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .exr files.")
    args = parser.parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        images_subdir=args.images_subdir,
        masks_subdir=args.masks_subdir,
        linearize=args.linearize_srgb,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
