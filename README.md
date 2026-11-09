# Low-Light Robust Finetuning of RoMa

This work presents a fine-tuned version of [RoMa](https://github.com/Parskatt/RoMa) designed for robust image matching under low-light conditions. The model is enhanced with degradation-aware training and advanced loss functions to improve robustness in challenging lighting scenarios.

## Overview

This project extends the original RoMa architecture by incorporating:
- **Degradation Simulation**: Low-light image degradation statistics computed via VDP-Net from [ELVIS](https://github.com/JoanneLin168/ELVIS/tree/main)
- **Robust Training**: Fine-tuning on [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/) dataset with degradation augmentation
- **Advanced Loss Functions**: InfoNCE and VICReg losses for improved matching robustness

## Setup

Environment setup and data processing follow the same procedures as the original [RoMa project](https://github.com/Parskatt/RoMa).

The degradation statistics for low-light video frames are precomputed and available in `./data/elvis.json`.

## Model Training

To train the model with degradation augmentation and pretrained weights:

```bash
python experiments/train_roma_outdoor.py \
    --is_degrade \
    --load_pretrain
```

**Parameters:**
- `--is_degrade`: Enable image degradation augmentation
- `--load_pretrain`: Load pretrained RoMa weights

**Finetuned model:**

Here we present our finetuned model in `./weights/RoMa_finetuned.pth`

## Model Evaluation

To evaluate the fine-tuned model:

```bash
python experiments/eval_roma_outdoor.py \
    --load_weight_path <path_to_your_weight>
```

Replace `<path_to_your_weight>` with the path to your trained model weights.

## Key Modifications

### 1. Data Degradation (`./romatch/datasets/megadepth.py`)

The degradation process simulates low-light conditions by adding realistic noise patterns to training images:

```python
# Degrade image with two different noise modes
if self.is_degrade:
    im_A1, mode1 = degradation.generate_noise_for_train(
        im_A, 
        self.degrade_params, 
        device=im_A.device, 
        use_normal=False
    )
    im_A2, mode2 = degradation.generate_noise_for_train(
        im_A, 
        self.degrade_params, 
        device=im_A.device, 
        use_normal=True
    )
```

### 2. Predictor Module (`./romatch/models/encoders.py`)

An additional predictor module has been integrated into the model architecture and is utilized in the `forward()` method of `RegressionMatcher` (`./romatch/models/matchers.py`).

### 3. Loss Functions (`./romatch/losses/robust_loss.py`)

Two advanced contrastive loss functions have been implemented:
- **InfoNCE Loss**: Contrastive learning approach for improved feature discrimination
- **VICReg Loss**: Variance-Invariance-Covariance regularization for stable training

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{li2026olivesoutdoorlowlightvideo,
      title={OLiVES: An Outdoor Low-Light Video Benchmark for Enhancement and Segmentation}, 
      author={Yini Li and Joanne Lin and David Blackham and David Bull and Nantheera Anantrasirichai},
      year={2026},
      eprint={2610.20501},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.20501}, 
}
```


## References

- [RoMa: Robust Matching](https://github.com/Parskatt/RoMa)
- [ELVIS: Enhance Low-Light for Video Instance Segmentation](https://github.com/JoanneLin168/ELVIS/tree/main)
- [MegaDepth Dataset](https://www.cs.cornell.edu/projects/megadepth/)
- [DKM: Dense Correspondence](https://github.com/Parskatt/DKM)
