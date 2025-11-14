# python
#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shutil
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Rename EXR frames from ####_*.exr to ####.exr safely.")
    parser.add_argument("folder", nargs="?", default=".", help="Target folder (default: current directory)")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what would be done, do not modify files")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Allow overwriting existing ####.exr")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    pat = re.compile(r"^(\d{4})_.+\.exr$", re.IGNORECASE)
    groups = {}  # frame -> list of matches

    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            frame = m.group(1)
            groups.setdefault(frame, []).append(p)

    if not groups:
        print(f"No files match pattern ####_*.exr in {folder}")
        sys.exit(1)

    # Preflight checks
    errors = 0
    for frame, files in sorted(groups.items(), key=lambda kv: int(kv[0])):
        if len(files) != 1:
            print(f"Frame {frame}: expected 1 match, found {len(files)}")
            for f in sorted(files):
                print(f" - {f.name}")
            errors += 1
            continue
        dst = folder / f"{frame}.exr"
        if dst.exists() and not args.overwrite:
            print(f"Destination exists (use --overwrite): {dst.name}")
            errors += 1

    if errors:
        print("Aborting due to the above issues.")
        sys.exit(2)

    # Apply renames
    for frame in sorted(groups.keys(), key=lambda x: int(x)):
        src = groups[frame][0]
        dst = folder / f"{frame}.exr"
        if args.dry_run:
            print(f"[DRY] {src.name} -> {dst.name}")
        else:
            if dst.exists():
                dst.unlink()  # only if --overwrite is set from preflight
            shutil.move(str(src), str(dst))
            print(f"Renamed {src.name} -> {dst.name}")

if __name__ == "__main__":
    main()
