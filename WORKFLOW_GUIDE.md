# OpenBias Complete Workflow Guide

This guide explains how to run the complete OpenBias pipeline from start to finish.

## Prerequisites

1. **Python 3.10** environment (already set up)
2. **Virtual environment activated**: `source openbias/bin/activate`
3. **Proposed biases file**: `proposed_biases/coco/3/coco_train.json` (already exists)
4. **Optional**: COCO dataset for original image comparison (if you want Step 3b)

## Complete Workflow Steps

### Step 1: Bias Proposal (Already Done ✓)
Generate bias proposals using an LLM (Llama2).

```bash
python bias_proposals.py --workers 6 --dataset 'coco'
```

**Output**: `proposed_biases/coco/3/coco_train.json`

**Note**: You already have this file, so you can skip this step.

---

### Step 2: Image Generation
Generate images using Stable Diffusion based on the proposed biases.

**For SD 1.5 (recommended for M1 Mac):**
```bash
python generate_images.py --dataset coco --generator sd-1.5
```

**For SD XL (higher quality, slower, more memory):**
```bash
python generate_images.py --dataset coco --generator sd-xl
```

**Output**: Images saved to `sd_generated_dataset/coco/train/{generator}/{caption_id}/`

**Configuration** (in `utils/config.py`):
- `'n-images': 50` - Number of images per caption (line 443)
- `'inference_steps': 20` - Generation steps (line 425)
- `'batch_size': 1` - Keep at 1 for MPS memory (line 424)

---

### Step 3a: VQA on Generated Images
Run Vision Question Answering on the generated images to detect biases.

```bash
python run_VQA.py --vqa_model blip-large --workers 1 --dataset 'coco' --mode 'generated' --generator sd-1.5
```

**Alternative VQA models** (if available):
- `blip-base` - Smaller, faster
- `blip-large` - Better accuracy (recommended)
- `llava-1.5-13b` - Requires model weights download
- `git-large` - Alternative option

**Output**:
- `results/VQA/coco/generated/{generator}/{vqa_model}/vqa_answers.json`
- `results/VQA/coco/generated/{generator}/{vqa_model}/data_counts.json`

---

### Step 3b: VQA on Original Images (Optional)
Run VQA on original COCO images for comparison.

**Prerequisites**:
- COCO dataset downloaded
- Update `utils/config.py` line 506: `'images_path': '/path/to/coco/images/train2017'`

```bash
python run_VQA.py --vqa_model blip-large --workers 1 --dataset 'coco' --mode 'original'
```

**Output**:
- `results/VQA/coco/original/{vqa_model}/vqa_answers.json`
- `results/VQA/coco/original/{vqa_model}/data_counts.json`

**Note**: Skip this step if you don't have the COCO dataset.

---

### Step 4: Plot Results
Generate visualization plots showing bias intensity.

**For generated images only:**
```bash
python make_plots.py --generator sd-1.5 --dataset coco --mode generated --vqa_model blip-large
```

**For original images (if Step 3b was run):**
```bash
python make_plots.py --dataset coco --mode original --vqa_model blip-large
```

**Output**:
- `results/VQA/coco/generated/{generator}/{vqa_model}/context_aware.png`
- `results/VQA/coco/generated/{generator}/{vqa_model}/context_free.png`

---

## Quick Start: Complete Pipeline (One Command Sequence)

```bash
# Activate environment
cd "/Users/nadaelsayed26/Documents/uni/year 1/3rd semester/Trends&App CV/openbias/OpenBias"
source openbias/bin/activate

# Step 1: Bias Proposal (skip if already done)
# python bias_proposals.py --workers 6 --dataset 'coco'

# Step 2: Generate Images
python generate_images.py --dataset coco --generator sd-1.5

# Step 3a: VQA on Generated Images
python run_VQA.py --vqa_model blip-large --workers 1 --dataset 'coco' --mode 'generated' --generator sd-1.5

# Step 3b: VQA on Original Images (optional, requires COCO dataset)
# python run_VQA.py --vqa_model blip-large --workers 1 --dataset 'coco' --mode 'original'

# Step 4: Plot Results
python make_plots.py --generator sd-1.5 --dataset coco --mode generated --vqa_model blip-large
```

---

## Configuration Tips

### For Faster Testing (Small Scale)
In `utils/config.py`:
- Set `'n-images': 2` (line 443) - Generate only 2 images per caption
- Set `'inference_steps': 20` (line 425) - Faster generation
- Use `sd-1.5` instead of `sd-xl` - Faster and less memory

### For Full Production Run
In `utils/config.py`:
- Set `'n-images': 50` (line 443) - Generate 50 images per caption
- Set `'inference_steps': 40` (line 425) - Higher quality
- Use `sd-xl` for better quality (if you have enough memory)

### Memory Optimization for M1 Mac
Already configured:
- `'batch_size': 1` - Prevents memory issues
- CPU offloading enabled for MPS
- Attention slicing enabled
- Float32 for MPS compatibility

---

## Expected Runtime (Approximate)

For **1 bias proposal** with **2 images**:
- Step 2 (Image Generation): ~2-4 minutes
- Step 3a (VQA): ~15-30 seconds
- Step 4 (Plotting): < 1 second

For **full dataset** (many bias proposals, 50 images each):
- Step 2: Several hours to days
- Step 3a: Several hours
- Step 4: < 1 minute

---

## Troubleshooting

### Out of Memory Errors
- Reduce `'n-images'` in config
- Use `sd-1.5` instead of `sd-xl`
- Reduce `'inference_steps'`
- Ensure CPU offloading is enabled (already done)

### VQA Model Not Found
- Use `blip-large` (auto-downloads from HuggingFace)
- For LLaVA, download weights to `utils/llava/weights/llava-v1.5-13b/`

### Missing Dependencies
- Run `pip install -r requirements.txt`
- Some models require additional packages (handled with lazy imports)

---

## Output Structure

```
OpenBias/
├── proposed_biases/
│   └── coco/3/
│       └── coco_train.json          # Step 1 output
├── sd_generated_dataset/
│   └── coco/train/sd-1.5/
│       └── {caption_id}/
│           └── 0.jpg, 1.jpg, ...    # Step 2 output
├── results/
│   └── VQA/
│       └── coco/
│           ├── generated/
│           │   └── sd-1.5/blip-large/
│           │       ├── vqa_answers.json    # Step 3a output
│           │       ├── data_counts.json     # Step 3a output
│           │       ├── context_aware.png    # Step 4 output
│           │       └── context_free.png     # Step 4 output
│           └── original/                    # Step 3b output (if run)
```

---

## Next Steps After Running

1. **Analyze the plots**: Check `context_free.png` and `context_aware.png` for bias intensity
2. **Compare with original**: If you ran Step 3b, compare generated vs original biases
3. **Scale up**: Increase `n-images` or add more bias proposals for comprehensive analysis
4. **Try different generators**: Compare `sd-1.5` vs `sd-xl` results
5. **Try different VQA models**: Compare `blip-large` vs other models

