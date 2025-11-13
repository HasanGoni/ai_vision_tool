# Notebooks

Interactive Jupyter notebooks for the AI Vision Tool.

## Available Notebooks

### 1. Synthetic Image Generation (`synthetic_image_generation.ipynb`)

Create synthetic images by composing pin objects with different backgrounds.

**Features:**
- Load pin images with optional masks
- Apply transformations (scale, rotation, flip, brightness, contrast)
- Composite pins onto various backgrounds
- Generate diverse training/testing data
- Save images with ground truth metadata
- Optional: Test with detection pipeline

**Use Cases:**
- Data augmentation for training
- Creating controlled test scenarios
- Evaluating detection performance
- Prototyping pin placements

**Quick Start:**
```bash
# Install Jupyter if not already installed
pip install jupyter notebook

# Start Jupyter
jupyter notebook

# Open synthetic_image_generation.ipynb
```

**Configuration:**
1. Set `PIN_IMAGE_PATH` to your pin image
2. Optionally provide `PIN_MASK_PATH` (or auto-generate)
3. Add background images to `BACKGROUND_PATHS`
4. Adjust transformation parameters
5. Run all cells to generate synthetic images

**Example Workflow:**
```python
# 1. Load pin
pin_img, pin_mask = load_pin_with_mask("data/pins/my_pin.jpg")

# 2. Load backgrounds
backgrounds = [cv2.imread(path) for path in background_paths]

# 3. Transform and composite
for bg in backgrounds:
    composite, metadata = composite_pin_on_background(
        bg, pin_img, pin_mask,
        position=None,  # Random
        blend_edges=True
    )

# 4. Save results
cv2.imwrite("output.jpg", composite)
```

## Data Directory Structure

```
data/
├── pins/                    # Pin images and masks
│   ├── my_pin.jpg
│   ├── my_pin_mask.png
│   └── ...
├── backgrounds/             # Background images
│   ├── outdoor_scene.jpg
│   ├── indoor_table.jpg
│   └── ...
└── synthetic_images/        # Generated synthetic images
    ├── synthetic_*.jpg
    └── synthetic_*.json     # Metadata with bounding boxes
```

## Requirements

The notebooks use the same dependencies as the main project. Install with:

```bash
pip install -r ../requirements.txt
pip install jupyter notebook
```

## Tips

### For Best Results:
1. **Pin Images**: Use high-quality images with clear edges
2. **Masks**: Provide masks for best compositing (or use clean white backgrounds)
3. **Backgrounds**: Choose diverse backgrounds matching your use case
4. **Lighting**: Match lighting conditions between pin and backgrounds
5. **Scale**: Adjust pin size to be realistic for the background

### Common Issues:

**Issue**: Pin edges look harsh
- **Solution**: Increase `EDGE_BLUR_RADIUS` for smoother blending

**Issue**: Auto-generated mask is poor
- **Solution**: Create a manual mask or use image editing tools

**Issue**: Pin doesn't fit in background
- **Solution**: Reduce `scale` parameter or use larger backgrounds

**Issue**: Synthetic images don't look realistic
- **Solution**: Adjust brightness/contrast to match background lighting

## Contributing

Feel free to add more notebooks for:
- Training data analysis
- Model evaluation
- Visualization tools
- Performance benchmarking
- Custom workflows

## Next Steps

1. Generate synthetic images with various backgrounds
2. Use synthetic data to test detection pipeline
3. Evaluate detection accuracy with known ground truth
4. Fine-tune detection thresholds based on results
5. Create training datasets for custom models
