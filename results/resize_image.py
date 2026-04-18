#!/usr/bin/env python3
"""
Resize images to match the format of plt.subplots(1, n_slices, figsize=(4, 4/2))

Usage:
    python resize_image.py --input "C:\path\to\image.png" --output "output.png"
    python resize_image.py --input "C:\path\to\folder" --n-slices 2
"""

import argparse
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def resize_image_to_subplot_format(input_path, output_path=None, n_slices=1, dpi=300):
    """
    Resize an image to match matplotlib subplot format.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image (optional)
        n_slices: Number of subplots (affects width calculation)
        dpi: DPI for conversion (default 300)
    """
    # Calculate target size in pixels
    # figsize=(4, 4/2) means width=4 inches, height=2 inches
    width_inches = 4
    height_inches = 4 / 2
    
    target_width = int(width_inches * dpi)
    target_height = int(height_inches * dpi)
    
    print(f"Target size: {target_width}x{target_height} pixels (at {dpi} DPI)")
    
    # Load image
    img = Image.open(input_path)
    original_size = img.size
    print(f"Original size: {original_size[0]}x{original_size[1]} pixels")
    
    # Resize image while maintaining aspect ratio
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Generate output path if not provided
    if output_path is None:
        input_pathlib = pathlib.Path(input_path)
        output_path = input_pathlib.parent / f"{input_pathlib.stem}_resized{input_pathlib.suffix}"
    
    # Save resized image
    img_resized.save(output_path, dpi=(dpi, dpi))
    print(f"Saved resized image to: {output_path}")
    
    return output_path


def process_folder(folder_path, n_slices=1, dpi=300, output_suffix="_resized"):
    """Process all images in a folder."""
    folder = pathlib.Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return
    
    # Supported image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {folder}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        output_path = img_path.parent / f"{img_path.stem}{output_suffix}{img_path.suffix}"
        try:
            resize_image_to_subplot_format(img_path, output_path, n_slices, dpi)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize images to match matplotlib subplot format (4x2 inches)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image file or folder path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=1,
        help="Number of subplots (affects width calculation)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for image conversion (default: 300)"
    )
    
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    
    if input_path.is_file():
        # Process single file
        resize_image_to_subplot_format(input_path, args.output, args.n_slices, args.dpi)
    elif input_path.is_dir():
        # Process folder
        process_folder(input_path, args.n_slices, args.dpi)
    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == "__main__":
    main()
