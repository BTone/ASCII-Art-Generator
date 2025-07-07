# ASCII Art Generator

A simple Python ASCII art generator that converts an image into ASCII art.

## Summary

Normalized bitmap masks for all printable ASCII characters from a given font are rendered at a given resolution.

Input images are split into non-overlapping tiles at the given resolution, and subsequently converted into normalized
grayscale bitmaps.

Least-squares fitting between the tile bitmaps and the normalized bitmaps is performed to find the best matching ASCII
character for each tile, which is then rendered as text.

Uses numpy for efficient vectorized operations.