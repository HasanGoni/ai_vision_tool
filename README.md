# AI Vision Tool: Custom Reference-Based Grounding

A powerful pipeline that combines **DINOv3**, **SAM2**, and **Grounding DINO** for custom object detection and segmentation.

## Overview

This tool allows you to teach AI to recognize and segment **your custom objects** by providing:
1. **Reference image** (visual example of what you're looking for)
2. **Text description** (semantic description in natural language)

### Perfect for:
- Detecting custom objects not in standard datasets
- Finding visually similar objects across images
- Teaching AI about domain-specific items (e.g., "my custom Pin design")
- Combining visual and semantic understanding for robust detection

## Architecture

```
Reference Image + Mask          Target Image + Text Prompt
         ↓                               ↓
    DINOv3 Features          →     Visual Similarity Map
         ↓                               ↓
                              Grounding DINO Detection
                                         ↓
                              Fusion (Visual + Text)
                                         ↓
                              SAM2 Segmentation
                                         ↓
                              Precise Instance Masks
```

### Components

1. **DINOv3** (Facebook Research)
   - Dense visual feature extraction
   - Finds regions similar to reference image
   - Self-supervised learning backbone

2. **Grounding DINO** (IDEA Research)
   - Open-vocabulary object detection
   - Detects objects using natural language descriptions
   - Enables teaching custom concepts via text

3. **SAM2** (Meta)
   - Segment Anything Model 2
   - Zero-shot instance segmentation
   - Precise mask generation

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 8GB+ VRAM

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ai_vision_tool
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Model-Specific Dependencies

#### SAM2
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

#### Grounding DINO
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

#### DINOv3
DINOv3 is loaded automatically via `torch.hub` on first use. Ensure you have internet connection.

### Step 5: Download Model Checkpoints (Optional)

Models will be downloaded automatically on first use. To pre-download:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# SAM2 checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_base_plus.pt -P checkpoints/

# Grounding DINO checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P checkpoints/
```

## Quick Start

### Example 1: Simple Reference Matching

```python
from pipelines.custom_grounding import CustomGroundingPipeline
from PIL import Image

# Initialize pipeline
pipeline = CustomGroundingPipeline()

# Load images
reference = Image.open("reference_red_car.jpg")
target = Image.open("street_scene.jpg")

# Extract reference features
ref_features = pipeline.process_reference_image(reference)

# Find similar objects
results = pipeline.detect_and_segment(
    target_image=target,
    reference_features=ref_features,
    similarity_threshold=0.5
)

print(f"Found {len(results['masks'])} similar objects")
```

### Example 2: Text-Based Custom Grounding

```python
# Detect your custom Pin using text description
results = pipeline.detect_and_segment(
    target_image=target,
    text_prompt="my custom Pin with red top and blue bottom",
    detection_threshold=0.3
)
```

### Example 3: Hybrid Approach (RECOMMENDED)

```python
# Combine visual similarity + text description for best results
results = pipeline.detect_and_segment(
    target_image=target,
    reference_features=ref_features,
    text_prompt="my custom Pin",
    fusion_strategy='multiply',  # Both signals must agree
    similarity_threshold=0.5,
    detection_threshold=0.3
)

# Visualize
vis = pipeline.visualize_results(target, results)
```

## Usage

### CLI Example

```bash
python ai_vision_tool/examples/custom_pin_detection.py \
  --reference data/reference/my_pin.jpg \
  --reference-mask data/reference/my_pin_mask.png \
  --target data/target/scene.jpg \
  --text-prompt "my custom Pin with unique pattern" \
  --output results/detection.jpg \
  --similarity-threshold 0.5 \
  --detection-threshold 0.3 \
  --fusion-strategy multiply
```

### Python API

See `ai_vision_tool/examples/simple_example.py` for more examples.

## Configuration

### Pipeline Parameters

```python
pipeline = CustomGroundingPipeline(
    dinov3_model="dinov2_vitb14",      # DINOv3 variant (vitb14, vitl14, vitg14)
    sam2_model="vit_b",                 # SAM2 variant (vit_b, vit_l, vit_h)
    grounding_dino_config=None,         # Custom config path
    device="cuda",                      # 'cuda' or 'cpu'
    use_visual_matching=True,           # Enable visual matching
    use_text_grounding=True             # Enable text grounding
)
```

### Detection Parameters

```python
results = pipeline.detect_and_segment(
    target_image=image,
    reference_features=ref_features,    # From process_reference_image()
    text_prompt="custom description",   # Natural language description
    fusion_strategy='multiply',         # 'multiply', 'max', or 'weighted'
    similarity_threshold=0.5,           # Visual similarity threshold (0-1)
    detection_threshold=0.3,            # Text detection threshold (0-1)
    min_region_area=100                 # Minimum mask area in pixels
)
```

### Fusion Strategies

1. **multiply** (Recommended): Both visual AND text must agree
   - Most precise, fewer false positives
   - Best when you have good reference image and clear text description

2. **max**: Either visual OR text can trigger detection
   - More permissive, higher recall
   - Good for exploratory detection

3. **weighted**: Weighted combination (60% visual, 40% text)
   - Balanced approach
   - Good for general use

## Use Cases

### 1. Custom Object Detection
Detect your specific objects that aren't in standard datasets:
```python
# Your custom Pin, badge, logo, product, etc.
results = pipeline.detect_and_segment(
    target_image=scene,
    reference_features=my_pin_features,
    text_prompt="my unique Pin design with custom logo"
)
```

### 2. Visual Similarity Search
Find all objects similar to a reference:
```python
# Find all red cars similar to reference
results = pipeline.detect_and_segment(
    target_image=parking_lot,
    reference_features=red_car_features,
    similarity_threshold=0.6
)
```

### 3. Semantic Object Grounding
Teach new concepts via text:
```python
# Detect using detailed description
results = pipeline.detect_and_segment(
    target_image=image,
    text_prompt="circuit board with blue capacitors and silver connectors"
)
```

### 4. Quality Control
Detect defects or specific patterns:
```python
# Find components matching specification
results = pipeline.detect_and_segment(
    target_image=product_image,
    reference_features=reference_good_component,
    text_prompt="component with correct orientation"
)
```

## Project Structure

```
ai_vision_tool/
├── models/
│   ├── dinov3.py              # DINOv3 wrapper
│   ├── sam2.py                # SAM2 wrapper
│   └── grounding_dino.py      # Grounding DINO wrapper
├── pipelines/
│   └── custom_grounding.py    # Main pipeline
├── utils/
│   ├── preprocessing.py       # Image preprocessing
│   ├── postprocessing.py      # Result post-processing
│   └── visualization.py       # Visualization utilities
├── examples/
│   ├── custom_pin_detection.py  # Full CLI example
│   └── simple_example.py        # Quick start examples
├── configs/
│   └── model_config.yaml      # Model configurations
└── data/                       # Sample data (add your own)
```

## Advanced Usage

### Using Custom Reference Masks

Focus feature extraction on specific object region:

```python
# Load reference image and mask
reference = load_image("pin.jpg")
mask = load_mask("pin_mask.png")

# Process with mask
ref_features = pipeline.process_reference_image(
    reference_image=reference,
    reference_mask=mask  # Only extract features from masked region
)
```

### Batch Processing

Process multiple images:

```python
import glob

target_images = glob.glob("data/targets/*.jpg")

for img_path in target_images:
    target = load_image(img_path)
    results = pipeline.detect_and_segment(
        target_image=target,
        reference_features=ref_features,
        text_prompt="my custom Pin"
    )
    save_results(f"results/{Path(img_path).stem}_result.jpg", target, results)
```

### Fine-tuning Detection

Adjust thresholds based on your needs:

```python
# High precision (fewer false positives)
results = pipeline.detect_and_segment(
    ...,
    similarity_threshold=0.7,    # Higher = stricter
    detection_threshold=0.4,     # Higher = stricter
    fusion_strategy='multiply'   # Both must agree
)

# High recall (catch more instances)
results = pipeline.detect_and_segment(
    ...,
    similarity_threshold=0.3,    # Lower = more permissive
    detection_threshold=0.2,     # Lower = more permissive
    fusion_strategy='max'        # Either can trigger
)
```

## Troubleshooting

### Out of Memory Error

1. Use smaller model variants:
```python
pipeline = CustomGroundingPipeline(
    dinov3_model="dinov2_vits14",  # Smaller variant
    sam2_model="vit_tiny"
)
```

2. Process on CPU:
```python
pipeline = CustomGroundingPipeline(device='cpu')
```

3. Resize images before processing:
```python
target = load_image("image.jpg", target_size=(800, 600))
```

### No Detections

1. Lower thresholds
2. Check if reference image is clear
3. Try different fusion strategy
4. Improve text description with more details

### Too Many False Positives

1. Increase thresholds
2. Use 'multiply' fusion strategy
3. Provide better reference image with mask
4. Make text description more specific

## Performance

Approximate inference time on RTX 3090:

- DINOv3 feature extraction: ~50ms per image
- Grounding DINO detection: ~100ms per image
- SAM2 segmentation: ~30ms per instance
- **Total**: ~200ms + (30ms × num_instances)

## Citation

If you use this tool in your research, please cite the original papers:

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv:2304.07193},
  year={2023}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv:2408.00714},
  year={2024}
}

@article{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and others},
  journal={arXiv:2303.05499},
  year={2023}
}
```

## License

This project uses models with different licenses:
- DINOv2: Apache 2.0
- SAM2: Apache 2.0
- Grounding DINO: Apache 2.0

Please check individual model repositories for specific license terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- Create an issue in the repository
- Check existing issues for solutions
- Refer to individual model documentation

## Acknowledgments

- Facebook Research for DINOv2 and SAM2
- IDEA Research for Grounding DINO
- The open-source computer vision community
