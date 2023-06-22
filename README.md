# Crop-You-Want-by-Text
A tool that allows you to crop multiple desired areas of multiple images based on [`Grounding DINO`](https://github.com/IDEA-Research/GroundingDINO).

## Install 

**Note:**

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

**Installation:**

Clone this repository from GitHub.

```bash
git clone https://github.com/orange-36/Crop-You-Want-by-Text.git
```

Change the current directory to the GroundingDINO folder.

```bash
cd Crop-You-Want-by-Text/GroundingDINO/
```

Install the required dependencies in the current directory.

```bash
pip install -e .
```

Download pre-trained model weights.

```bash
cd ../..
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## Getting Started
Execute the program and give a text prompt to cut out the desired part of the image.
```bash
python crop_you_want.py --image_path <image_path> --text_prompt "<text_prompt>"
```

Parameter:
* `--image_path` The input image path, you can input multiple paths or the directory path containing images to be processed.
* `--text_prompt` Enter a text prompt, which can be a word or a phrase. Use `.` to separate different text categories.
* `--box_threshold` Threshold for bounding box. (default: 0.25)
* `--text_threshold` Threshold to judge whether it is the corresponding text category. (default: 0.25)
* `--extend` Extra dilated target box to crop. (default: 0)
* `--model_config` [`Model config`](https://github.com/IDEA-Research/GroundingDINO/tree/9389fa492b0188ab85d2bba902f5451c0b1528d1/groundingdino/config) used by GroundingDINO. (defalt: `"groundingdino/config/GroundingDINO_SwinT_OGC.py"`) 
* `--model_weight` [`Pretrained model weights`](https://github.com/IDEA-Research/GroundingDINO/tree/9389fa492b0188ab85d2bba902f5451c0b1528d1#luggage-checkpoints) used by GroundingDINO. (defalt: `"weights/groundingdino_swint_ogc.pth"`)
* `--output_path` Where to save the results. (default: `"output/"`)
* `--device` Device want to use. If no gpu is available, set `"cpu"`. (default: `"cuda"`)
* `--output_order` The order of the output results, set `score` to go from high to low according to the score, or set `x` or `y` to go from left to right or top to bottom. (default: `score`)
* `--no_sub_dir` Do not create additional subdirectories, take effect by entering `--no_sub_dir`.
* `--square_crop` Crop the image into a square, take effect by set `--square_crop`.

## Citation

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```