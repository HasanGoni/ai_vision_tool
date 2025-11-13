# AI Vision Tool: DINOv2 + SAM2 Integration

🚀 **Automatic Object Segmentation** combining DINOv2 detection with SAM2 segmentation

## Overview

This project implements **automatic object segmentation** by combining two powerful vision models:

1. **DINOv2** (Meta): Detects salient objects and regions automatically
2. **SAM2** (Meta): Performs precise segmentation based on DINO's detections

### The Problem with Standard SAM2

Standard SAM2 requires **manual prompts** (clicks, boxes, etc.) to segment objects:
```
Image → [👆 Manual prompt] → SAM2 → Segmentation
```

### Our Solution: Automatic Prompting

We use **DINOv2 to automatically generate prompts** for SAM2:
```
Image → DINOv2 (detects objects) → Auto prompts → SAM2 → Segmentation ✅
```

**No manual input required!** Just provide an image and get segmented objects.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: DINOv2 Object Detection                           │
│  ─────────────────────────────────                          │
│  • Extract dense visual features                            │
│  • Compute attention maps                                   │
│  • Detect salient regions automatically                     │
│  • Generate bounding boxes and center points                │
└────────────────────┬────────────────────────────────────────┘
                     │ Detected Regions
                     │ (boxes, points, confidence)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: SAM2 Segmentation                                 │
│  ───────────────────────────                                │
│  • Use DINO regions as prompts                              │
│  • Generate precise pixel-level masks                       │
│  • Refine segmentation boundaries                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: Segmented Objects                                   │
│  • Individual masks for each object                          │
│  • Bounding boxes and confidence scores                      │
│  • Visualizations                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

✨ **Fully Automatic**: No manual prompts needed
🎯 **High Precision**: Combines DINO's detection with SAM2's segmentation quality
⚡ **Flexible**: Supports both bounding box and point prompts
📊 **Configurable**: Adjust detection sensitivity, number of objects, etc.
💾 **Complete Output**: Saves masks, visualizations, and metadata
🔄 **Batch Processing**: Process multiple images efficiently

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ai_vision_tool
```

### 2. Install dependencies

```bash
# Install PyTorch (choose your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install other requirements
pip install -r requirements.txt
```

### 3. Download SAM2 checkpoint (optional)

```bash
# Download SAM2 checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

**Note**: DINOv2 models download automatically via torch.hub on first use.

---

## Quick Start

### Basic Usage

```python
from src.dino_sam_pipeline import DinoSamPipeline

# Initialize pipeline
pipeline = DinoSamPipeline()

# Process image - that's it!
results = pipeline.process_image(
    image_path="your_image.jpg",
    save_dir="output"
)
```

### Command Line

```bash
# Process single image
python example_usage.py image.jpg --save-dir output

# Process multiple images
python example_usage.py image1.jpg image2.jpg image3.jpg

# Customize detection
python example_usage.py image.jpg \
    --num-objects 10 \
    --threshold 0.5 \
    --prompt-type box \
    --dino-model dinov2_vitl14
```

### Run Demo

```bash
# Runs automatic demo with test image
python example_usage.py
```

---

## Usage Examples

### Example 1: Basic Segmentation

```python
from src.dino_sam_pipeline import DinoSamPipeline

pipeline = DinoSamPipeline()

results = pipeline.process_image(
    image_path="photo.jpg",
    num_objects=5,           # Detect up to 5 objects
    prompt_type="box",       # Use bounding boxes
    save_dir="output"
)

print(f"Found {len(results['segmentations'])} objects")
```

### Example 2: Fine-Tuned Detection

```python
# More sensitive detection
results = pipeline.process_image(
    image_path="photo.jpg",
    num_objects=10,
    attention_threshold=0.5,  # Lower = more sensitive
    min_area=50,              # Smaller minimum size
    prompt_type="point"       # Use point prompts
)
```

### Example 3: Batch Processing

```python
from pathlib import Path

# Get all images
image_paths = list(Path("images/").glob("*.jpg"))

# Process all at once
results = pipeline.process_batch(
    image_paths=[str(p) for p in image_paths],
    save_dir="output/batch"
)
```

### Example 4: Programmatic Access

```python
results = pipeline.process_image("photo.jpg", save_dir=None)

# Access results
for idx, seg in enumerate(results['segmentations']):
    mask = seg['mask']                    # Binary mask
    bbox = seg['bbox']                    # Bounding box
    sam_score = seg['score']              # SAM2 confidence
    dino_conf = seg['confidence']         # DINO confidence

    # Extract object
    object_crop = results['image'][mask]

    # Your custom processing...
```

---

## Configuration Options

### DINOv2 Models

Choose model size (larger = better but slower):

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `dinov2_vits14` | Small | ⚡⚡⚡ | ⭐⭐ |
| `dinov2_vitb14` | Base | ⚡⚡ | ⭐⭐⭐ |
| `dinov2_vitl14` | Large | ⚡ | ⭐⭐⭐⭐ |
| `dinov2_vitg14` | Giant | 🐌 | ⭐⭐⭐⭐⭐ |

### Detection Parameters

```python
pipeline.process_image(
    image_path="...",

    # Detection settings
    num_objects=5,              # Max objects to detect
    attention_threshold=0.6,    # 0.0-1.0, lower = more sensitive
    min_area=100,               # Minimum region size (pixels)

    # Prompt type
    prompt_type="box",          # "box" or "point"

    # Output settings
    visualize=True,             # Create visualizations
    save_dir="output"           # Where to save results
)
```

### Recommended Settings by Use Case

**General purpose**:
```python
num_objects=5, threshold=0.6, prompt_type="box"
```

**Many small objects** (e.g., crowd photos):
```python
num_objects=20, threshold=0.5, min_area=50, prompt_type="point"
```

**Few large objects** (e.g., vehicles):
```python
num_objects=3, threshold=0.7, min_area=500, prompt_type="box"
```

**Maximum sensitivity** (detect everything):
```python
num_objects=50, threshold=0.4, min_area=20
```

---

## Output Structure

```
output/
├── image_name_visualization.jpg       # Side-by-side comparison
├── image_name_summary.txt             # Detection statistics
└── image_name_masks/
    ├── mask_000.png                   # Binary mask for object 1
    ├── mask_000.txt                   # Metadata for object 1
    ├── mask_001.png                   # Binary mask for object 2
    ├── mask_001.txt                   # Metadata for object 2
    └── ...
```

### Visualization Panels

The visualization image contains three panels:
1. **Original**: Input image
2. **DINO Detections**: Attention maps + bounding boxes
3. **SAM2 Segments**: Final segmentation masks

---

## How It Works

### Stage 1: DINOv2 Detection

1. **Feature Extraction**: DINOv2 extracts dense visual features
2. **Attention Map**: Compute self-attention to find salient regions
3. **Region Detection**: Find connected components in attention map
4. **Prompt Generation**: Extract bounding boxes and center points

### Stage 2: SAM2 Segmentation

1. **Prompt Input**: Use DINO regions as prompts
2. **Segmentation**: SAM2 generates precise pixel masks
3. **Refinement**: SAM2's boundary refinement produces clean masks

### Why This Works

- **DINOv2** is trained on diverse data and naturally attends to objects
- **SAM2** excels at precise segmentation when given good prompts
- **Combination** leverages strengths of both models

---

## Advanced Usage

### Custom DINOv2 Processing

```python
from src.dino_detector import DINOv2Detector

detector = DINOv2Detector(model_name="dinov2_vitl14")

# Extract features
features = detector.extract_features(image)

# Detect regions
regions = detector.detect_salient_regions(
    image,
    num_regions=10,
    attention_threshold=0.5
)

# Visualize
vis = detector.visualize_detections(image, regions)
```

### Custom SAM2 Processing

```python
from src.sam2_segmenter import SAM2Segmenter

segmenter = SAM2Segmenter()
segmenter.set_image(image)

# Segment from boxes
boxes = np.array([[x1, y1, x2, y2], ...])
results = segmenter.segment_from_boxes(boxes)

# Segment from points
points = np.array([[x, y], ...])
results = segmenter.segment_from_points(points)
```

---

## API Reference

### DinoSamPipeline

**Main class for automatic segmentation**

```python
pipeline = DinoSamPipeline(
    dino_model="dinov2_vitb14",
    sam2_config="sam2_hiera_l.yaml",
    sam2_checkpoint=None,
    device=None  # Auto-detect
)
```

**Methods**:
- `process_image(image_path, **kwargs)` - Process single image
- `process_batch(image_paths, **kwargs)` - Process multiple images

### DINOv2Detector

**Object detection using DINOv2**

```python
detector = DINOv2Detector(model_name="dinov2_vitb14")
```

**Methods**:
- `extract_features(image)` - Extract DINO features
- `detect_salient_regions(image, ...)` - Detect objects
- `visualize_detections(image, regions)` - Create visualization

### SAM2Segmenter

**Segmentation using SAM2**

```python
segmenter = SAM2Segmenter()
```

**Methods**:
- `set_image(image)` - Set image to segment
- `segment_from_boxes(boxes)` - Segment from box prompts
- `segment_from_points(points)` - Segment from point prompts
- `segment_from_regions(regions)` - Segment from DINO regions
- `visualize_masks(image, results)` - Visualize masks

---

## Troubleshooting

### "No regions detected"

Try lowering `attention_threshold`:
```python
results = pipeline.process_image(image, attention_threshold=0.4)
```

### "Too many false detections"

Try raising `attention_threshold` or `min_area`:
```python
results = pipeline.process_image(
    image,
    attention_threshold=0.7,
    min_area=200
)
```

### "Out of memory"

Use a smaller DINOv2 model:
```python
pipeline = DinoSamPipeline(dino_model="dinov2_vits14")
```

### "SAM2 not loading"

Check SAM2 installation:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

---

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Manual SAM2** | Highest control | Requires manual prompts |
| **Grounding DINO + SAM2** | Text-based control | Needs text descriptions |
| **DINOv2 + SAM2** (Ours) | Fully automatic | Less control over what to segment |
| **End-to-end models** | Single model | Often less accurate |

---

## Future Improvements

- [ ] Support for video/multi-frame processing
- [ ] Few-shot object matching (segment similar objects)
- [ ] Integration with Grounding DINO for text prompts
- [ ] Real-time processing optimizations
- [ ] Web interface for easy use

---

## Citation

If you use this work, please cite the underlying models:

**DINOv2**:
```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

**SAM2**:
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

---

## License

This project is for research and educational purposes. Please respect the licenses of DINOv2 and SAM2.

---

## Acknowledgments

- Meta AI for DINOv2 and SAM2
- PyTorch team for the excellent framework
- Open source community

---

## Questions?

Feel free to open an issue or reach out!

**Your idea was excellent - combining DINO's automatic detection with SAM2's precise segmentation gives you the best of both worlds! 🎉**
