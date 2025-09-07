
Core Steps of the Program:

1. Paper Extraction – Capture an image of the paper; the paper’s dimensions must be known in advance.
2. Unwrap / Straighten – Process the image to flatten the paper surface.
3. Dilate Contour – Expand the hand’s contour by a specified amount (e.g., 5 mm).
4. Export – Convert the result to a printable PDF or PNG at the chosen paper size.

> Tip: Use a larger sheet of paper, because dilation increases the required area.

> Note: The output includes a 10 cm vertical line and a 10 cm horizontal line so you can verify that the printed size isn’t distorted. This program assumes minimal camera distortion.

# Installation

1. Install conda/mamba/micromamba
2. `cd /path/to/this/repo`
3. `conda create -f envieonment.yaml`
4. `conda activate -n fs_ptrn`
