import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CHARACTERS = "".join(chr(i) for i in range(32, 127))  # Printable ASCII characters


def generate_glyphs(truetype_font_path, character_set=_CHARACTERS, font_size=16):
    font = ImageFont.truetype(
        truetype_font_path, font_size, layout_engine=ImageFont.Layout.BASIC
    )

    chars = []
    glyphs = []
    glyph_masks = []
    densities = []

    for char in character_set:
        # Create glyph
        im = Image.new("L", (font_size // 2, font_size), 0)  # Create a black image
        draw = ImageDraw.Draw(im)
        draw.fontmode = "1"
        draw.text((0, 0), char, font=font, fill=255)

        # Get mask and compute density
        mask = (np.array(im) / 255.0).flatten()
        density = (np.array(im)).mean()

        # Append to lists
        chars.append(char)
        glyphs.append(im)
        glyph_masks.append(mask)
        densities.append(density)

    # Sort entires by density
    sorted_lists = sorted(
        zip(chars, glyphs, glyph_masks, densities, strict=True), key=lambda x: x[3]
    )

    chars, glyphs, glyph_masks, densities = zip(*sorted_lists)

    # Concatenate glyph masks
    glyph_masks = np.array(glyph_masks)

    return chars, glyphs, glyph_masks, np.array(densities)


def tile_image(image, tile_size=(8, 16)):
    tile_size_x, tile_size_y = tile_size
    width, height = image.size

    num_tiles_x = width // tile_size_x
    num_tiles_y = height // tile_size_y

    tiles = []
    normalized_tiles = []

    for j in range(0, num_tiles_y):
        for i in range(0, num_tiles_x):
            box = (
                i * tile_size_x,
                j * tile_size_y,
                (i + 1) * tile_size_x,
                (j + 1) * tile_size_y,
            )
            tile = image.crop(box)
            normalized_tile = (np.array(tile) / 255.0).flatten()

            tiles.append(tile)
            normalized_tiles.append(normalized_tile)

    return (num_tiles_x, num_tiles_y), tiles, np.array(normalized_tiles)


def compute_error(normalized_tiles, glyph_masks):
    reshaped_glyph_masks = np.swapaxes(np.expand_dims(glyph_masks, axis=2), 0, 2)
    reshaped_normalized_tiles = np.expand_dims(normalized_tiles, axis=2)

    # L2 norm
    error = (reshaped_normalized_tiles - reshaped_glyph_masks) ** 2
    error = np.sum(error, axis=1)  # Sum over the tile dimension

    return error


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="ASCII Art Generator")
    parser.add_argument(
        "input_image",
        type=Path,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--font_path",
        type=Path,
        default=Path("C:/Windows/Fonts/consola.ttf"),
        help="Path to the TrueType font file",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=(6, 12),
        help="Size of the tiles (width, height)",
    )
    args = parser.parse_args()
    if not args.input_image.is_file():
        print(
            f"Error: Input image '{args.input_image}' does not exist.", file=sys.stderr
        )
        sys.exit(1)

    tile_size_x, tile_size_y = args.tile_size
    characters, glyphs, glyph_masks, densities = generate_glyphs(
        args.font_path, font_size=tile_size_y
    )

    input = Image.open(args.input_image).convert("L")
    (num_tiles_x, num_tiles_y), tiles, normalized_tiles = tile_image(
        input, (tile_size_x, tile_size_y)
    )
    error = compute_error(normalized_tiles, glyph_masks)
    output = [characters[i] for i in error.argmin(axis=1)]
    for i in range(num_tiles_y):
        row = output[i * num_tiles_x : (i + 1) * num_tiles_x]
        print("".join(row))
