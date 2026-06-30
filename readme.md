
# <img src="assets/turtle.png" alt="Lego Turtle" width="50"> **Turtle: Learning Truncated Causal History Model for Video Restoration [NeurIPS'2024]**

[📄 arxiv](https://arxiv.org/abs/2410.03936)
**|** 
[🌐 Website](https://kjanjua26.github.io/turtle/)

The official PyTorch implementation for **Learning Truncated Causal History Model for Video Restoration**, accepted to NeurIPS 2024. 

- Turtle achieves state-of-the-art results on multiple video restoration benchmarks, offering superior computational efficiency and enhanced restoration quality 🔥🔥🔥.
- **🛠️💡Model Forge**: Easily design your own architecture by modifying the option file.
    - You have the flexibility to choose from various types of layers—such as channel attention, simple channel attention, CHM, FHR, or custom blocks—as well as different types of feed-forward layers.
    - This setup allows you to create custom networks and experiment with layer and feed-forward configurations to suit your needs. 
- If you like this project, please give us a ⭐ on Github!🚀 

<p align="center">
  <img src="assets/gopro.gif" alt="Restored Video 1" width="280" height="180">
  <img src="assets/nightrain30.gif" alt="Restored Video 2" width="280" height="180">
</p>

<p align="center">
  <img src="assets/raindrop.gif" alt="Restored Video 3" width="280" height="180">
  <img src="assets/snowwww.gif" alt="Restored Video 4" width="280" height="180">
</p>

### 🔥 📰 News 🔥
- Oct. 10, 2024: The paper is now available on [arxiv](http://export.arxiv.org/abs/2410.03936) along with the code and pretrained models.
- Sept 25, 2024: Turtle is accepted to NeurIPS'2024.

  
## Table of Contents
1. [Quickstart with uv](#quickstart-with-uv-recommended)
2. [Real-World Deblurring (run it now)](#real-world-deblurring-run-it-now)
3. [Installation](#installation)
4. [Trained Models](#trained-models)
5. [Dataset Preparation](#1-dataset-preparation)
6. [Training](#2-training)
7. [Evaluation](#3-evaluation)
    - [Testing the Model](#31-testing-the-model)
    - [Inference on Given Videos](#32-inference-on-given-videos)
8. [Model Complexity and Inference Speed](#4-model-complexity-and-inference-speed)
9. [Acknowledgments](#5-Acknowledgments)
10. [Citation](#6-citation)


## Quickstart with uv (recommended)

This repo ships with a [uv](https://docs.astral.sh/uv/) project (`pyproject.toml`) so
you can create a reproducible environment in one command. The pinned build uses
**PyTorch 2.4.1 + CUDA 12.1**, which runs on modern GPUs (RTX 30/40-series, Ada, etc.)
as well as on CPU.

```bash
# 1. Install uv (if you don't have it):
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install all dependencies:
cd Turtle
uv sync

# 3. Sanity-check the install (prints torch version + GPU):
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`uv sync` creates a local `.venv/`. Prefix any command with `uv run` to use it
(e.g. `uv run python run_realworld_deblur.py ...`), or activate it with
`source .venv/bin/activate`.

> **Note:** Unlike the original instructions, you do **not** need to run
> `python setup.py develop`. `basicsr` is imported directly from the source tree,
> so `uv sync` is all that is required.

### Download the pretrained models

All checkpoints live in a single Google Drive folder. Download them with `gdown`
(installed by `uv sync`) into `trained_models/`:

```bash
uv run gdown --folder "https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA" -O trained_models
```

This fetches every task's model. The file used for **real-world deblurring** is
`BSD_Deblur.pth`. The helper script below defaults to `trained_models/BSD.pth`, so
either copy it or pass `--model trained_models/BSD_Deblur.pth`:

```bash
cp trained_models/BSD_Deblur.pth trained_models/BSD.pth
```


## Real-World Deblurring (run it now)

`run_realworld_deblur.py` is a clean, self-contained entry point for the
real-world deblurring task (BSD model). It takes a folder of ordered video frames
and writes the deblurred frames to an output folder. Frames are processed
**sequentially**, so Turtle's truncated causal history (the K/V cache) carries
across the whole sequence.

```bash
uv run python run_realworld_deblur.py \
    --input  datasets/your_video_frames \
    --output outputs/realworld_deblur \
    --model  trained_models/BSD.pth
```

| flag             | default                            | description                                   |
| ---------------- | ---------------------------------- | --------------------------------------------- |
| `--input`        | *(required)*                       | Folder with ordered frames (`*.png/*.jpg`)    |
| `--output`       | `outputs/realworld_deblur`         | Where deblurred frames are written            |
| `--model`        | `trained_models/BSD.pth`           | Path to the real-world deblurring checkpoint  |
| `--config`       | `options/Turtle_Derain_VRDS.yml`   | Network/architecture option file              |
| `--model-type`   | `t0`                               | Turtle variant used by this checkpoint        |
| `--tile`         | `320`                              | Tile size (multiple of 8); lowers VRAM use    |
| `--tile-overlap` | `128`                              | Overlap between tiles                          |
| `--device`       | `auto`                             | `auto` / `cuda` / `cpu`                        |

**Have only a video file?** Extract frames first, then run the script:

```bash
# edit video_path / output_folder inside the script, then:
uv run python video_to_frames.py
uv run python run_realworld_deblur.py --input <output_folder> --output outputs/realworld_deblur
```

The deblurred frames can be stitched back into a video with `make_video.py`.


## Restore a sequence of images — any task

`restore_sequence.py` is the general entry point: give it a folder of numbered
frames and a task preset, and it restores the whole sequence (frames are
natural-sorted, so `frame2.png` comes before `frame10.png`). It covers every task
that has a public checkpoint and can optionally assemble the result into an mp4.

```bash
# see all task presets:
uv run python restore_sequence.py --list

# restore a sequence (e.g. real-world deblurring) and also build a video:
uv run python restore_sequence.py \
    --task deblur-realworld \
    --input  datasets/my_frames \
    --output outputs/restored \
    --video --fps 24
```

Task presets (each maps to the right config / checkpoint / model variant):

| `--task`           | checkpoint (`trained_models/`) | model_type |
| ------------------ | ------------------------------ | ---------- |
| `deblur-realworld` | `BSD_Deblur.pth`               | `t0`       |
| `deblur-gopro`     | `GoPro_Deblur.pth`             | `t1`       |
| `desnow`           | `Desnow.pth`                   | `t0`       |
| `derain-night`     | `NightRain.pth`                | `t0`       |
| `raindrop`         | `RainDrop.pth`                 | `t1`       |
| `sr`               | `SuperResolution.pth`          | `SR`       |

Useful flags:

| flag                  | description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `--input`             | Folder of ordered frames (`.png/.jpg/.jpeg/.bmp/.tif`)        |
| `--output`            | Output folder (default `outputs/restored`)                    |
| `--model` / `--config`| Override the preset's checkpoint / option file                |
| `--tile` / `--tile-overlap` | Tiling (lower `--tile` if you run out of VRAM)          |
| `--no-patches`        | Process whole frames at once (needs more VRAM)                |
| `--device`            | `auto` / `cuda` / `cpu`                                       |
| `--video` / `--fps`   | Also write `<task>.mp4` from the restored frames              |

> `run_realworld_deblur.py` (above) is a thin, deblur-only convenience wrapper;
> `restore_sequence.py` is the general tool and is recommended for all tasks.

> Both scripts are dataset-agnostic: all paths (input/output/checkpoint) are passed
> on the command line, so no private dataset names or paths are hard-coded inside.


## Side-by-side comparison video

`make_comparison_video.py` builds a comparison video from two frame folders
(e.g. original on the left, restored on the right). It is a generic tool — it only
takes folder paths — and shows a tqdm progress bar.

```bash
uv run python make_comparison_video.py \
    --left  input_frames \
    --right outputs/restored \
    --output comparison.mp4 \
    --fps 24 --labels "Original,Restored"
```

| flag       | description                                              |
| ---------- | -------------------------------------------------------- |
| `--left`   | Left-pane frames (e.g. the original input)               |
| `--right`  | Right-pane frames (e.g. the restored output)             |
| `--output` | Output mp4 path                                          |
| `--fps`    | Frames per second                                        |
| `--labels` | `"left,right"` captions (pass empty to disable)          |
| `--gap`    | White gap (px) between the two panes                     |

Frames are paired in natural-sort order; if the right frames have a different
height they are resized to match the left.

### Pre-processing frames with ffmpeg (optional)

If your frames need to be rotated/scaled first, do it with ffmpeg before restoration,
e.g. rotate every frame 90° counter-clockwise into a new folder:

```bash
ffmpeg -start_number 1 -i 'rgb/%08d.jpg' -vf transpose=2 'rgb_ccw90/%08d.png'
# transpose=1 -> 90° clockwise, transpose=2 -> 90° counter-clockwise
```


## Installation

This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR) which is an open-source toolbox for image/video restoration tasks.

The recommended setup is [Quickstart with uv](#quickstart-with-uv-recommended) above.
If you prefer a manual `pip` install (the configuration the paper was originally
developed with), use:

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```bash
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

> On newer GPUs (Ada / RTX 40-series, sm_89), PyTorch 1.11 will fail at runtime —
> use the uv environment (PyTorch 2.4.1 + CUDA 12.1) instead.

## Trained Models

You can download our trained models from Google Drive: [Trained Models](https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA?usp=sharing)

| Task                   | Checkpoint            | `dataset_name` | `task_name`        | `model_type` | config                          |
| ---------------------- | --------------------- | -------------- | ------------------ | ------------ | ------------------------------- |
| Real-World Deblurring  | `BSD_Deblur.pth`      | `BSD`          | `Deblurring`       | `t0`         | `options/Turtle_Derain_VRDS.yml`|
| Synthetic Deblurring   | `GoPro_Deblur.pth`    | `GoPro`        | `Deblurring`       | `t1`         | `options/Turtle_Deblur_Gopro.yml`|
| Desnowing              | `Desnow.pth`          | `RSVD`         | `Desnowing`        | `t0`         | `options/Turtle_Desnow.yml`     |
| Night Deraining        | `NightRain.pth`       | `NightRain`    | `Deraining`        | `t0`         | `options/Turtle_Derain.yml`     |
| Raindrop Removal       | `RainDrop.pth`        | `VRDS`         | `Deraining`        | `t1`         | `options/Turtle_Derain_VRDS.yml`|
| Super-Resolution       | `SuperResolution.pth` | `MVSR`         | `SuperResolution`  | `SR`         | `options/Turtle_SR_MVSR.yml`    |


## 1. Dataset Preparation
To obtain the datasets, follow the official instructions provided by each dataset's provider and download them into the dataset folder. You can download the datasets for each of the task from the following links (official sources reported by their respective authors).

1. Desnowing: [RSVD](https://haoyuchen.com/VideoDesnowing)
2. Raindrops and Rainstreaks Removal: [VRDS](https://hkustgz-my.sharepoint.com/personal/hwu375_connect_hkust-gz_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhwu375%5Fconnect%5Fhkust%2Dgz%5Fedu%5Fcn%2FDocuments%2FVRDS&ga=1)
3. Night Deraining: [NightRain](https://drive.google.com/drive/folders/1zsW1D8Wtj_0GH1OOHSL7dwR_MIkZ8-zp?usp=sharing)
4. Synthetic Deblurring: [GoPro](https://seungjunnah.github.io/Datasets/gopro)
5. Real-World Deblurring: [BSD3ms-24ms](https://drive.google.com/drive/folders/1LKLCE_RqPF5chqWgmh3pj7cg-t9KM2Hd?usp=sharing)
6. Denoising: [DAVIS](https://github.com/m-tassano/fastdvdnet?tab=readme-ov-file) | [Set8](https://drive.google.com/drive/folders/11chLkbcX-oKGLOLONuDpXZM2-vujn_KD?usp=sharing)
7. Real-World Super Resolution: [MVSR](https://github.com/HITRainer/EAVSR?tab=readme-ov-file)

The directory structure, including the ground truth ('gt') for reference frames and 'blur' for degraded images, should be organized as follows:

```bash
./datasets/
└── Dataset_name/
    ├── train/
    └── test/
        ├── blur
           ├── video_1
           │   ├── Fame1
           │   ....
           └── video_n
           │   ├── Fame1
           │   ....
        └── gt
           ├── video_1
           │   ├── Fame1
           │   ....
           └── video_n
           │   ├── Fame1
           │   ....
```

## 2. Training
To train the model, make sure you select the appropriate data loader in the `train.py`. There are two options as follows.

1. For deblurring, denoising, deraining, etc. keep the following import line, and comment the superresolution one.
`from basicsr.data.video_image_dataset import VideoImageDataset` 

2. For superresolution, keep the following import line, and comment the previous one.
`from basicsr.data.video_super_image_dataset import VideoSuperImageDataset as VideoImageDataset`

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8080 basicsr/train.py -opt /options/option_file_name.yml --launcher pytorch
```

## 3. Evaluation

The pretrained models can be downloaded from the [GDrive link](https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA?usp=sharing).

### 3.1 Testing the model
To evaluate a pre-trained model on a benchmark **with ground truth** (computing PSNR/SSIM),
use `basicsr/inference.py`:

```bash
uv run python basicsr/inference.py
```

> **Note:** `basicsr/inference.py` is configured by editing the `__main__` block at the
> bottom of the file (uncomment the task you want and set the paths). The default paths use
> a leading `/` (e.g. `/datasets/`, `/outputs/`) that assume a container layout — change
> `pth_to_dataset_folder` and `image_out_path` to your local paths first. For real-world
> deblurring without ground truth, prefer [`run_realworld_deblur.py`](#real-world-deblurring-run-it-now).

Adjust the function parameters in the Python file according to each task requirements:
1. `config`: Specify the path to the option file.
2. `model_path`: Provide the location of pre-trained model.
3. `dataset_name`: Select the dataset you are using ("RSVD", "GoPro", "SR", "NightRain", "DVD", "Set8").
4. `task_name`: Choose the restoration task ("Desnowing", "Deblurring", "SR", "Deraining", "Denoising").
5. `model_type`: Indicate the model type ("t0", "t1", "SR").
6. `save_image`: Set to `True` if you want to save the output images; provide the output path in `image_out_path`.
7. `do_patches`: Enable if processing images in patches; adjust `tile` and `tile_overlap` as needed, default values are 320 and 128.
8. `y_channel_PSNR`: Enable if need to calculate PSNR/SSIM in Y Channel, default is set to False.


### 3.2 Running Turtle on Custom Videos:

This pipeline processes a video by extracting frames and running a pre-trained model for tasks like desnowing:

#### Step 1: Extract Frames from Video

1. Edit `video_to_frames.py`:
   - Set the `video_path` to your input video file.
   - Set the `output_folder` to save extracted frames.

2. Run the script:
   ```bash
   uv run python video_to_frames.py
   ```

#### Step 2: Run Model Inference

For **real-world deblurring**, simply point the helper script at the extracted frames
(see [Real-World Deblurring](#real-world-deblurring-run-it-now)):

```bash
uv run python run_realworld_deblur.py --input <extracted_frames> --output outputs/realworld_deblur
```

For other tasks you can use the lower-level `basicsr/inference_no_ground_truth.py`
(edit `config`, `model_path`, `data_dir`, and `image_out_path` in its `__main__` block, then
`uv run python basicsr/inference_no_ground_truth.py`).


## 4. Model complexity and inference speed
* To get the parameter count, MAC, and inference speed use this command:
```bash
uv run python basicsr/models/archs/turtle_arch.py
```

### Contributions 📝📝

We invite the community to contribute to extending **TURTLE** to other low-level vision tasks. Below is a list of specific areas where contributions could be highly valuable if the models are open-sourced. If you have other suggestions or requests, please feel free to open an issue.

1. **Training TURTLE for Synthetic Super-Resolution Tasks**  
   - **Bicubic (BI) Degradation**: Train on REDS, Vimeo90K and evaluate on REDSS4, Vimeo90K-T.  
   - **Blur-Downsampling (BD) Degradation**: Train on Vimeo90K and evaluate on Vimeo90K-T, Vid4, UDM10.

For more information on dataset selection and data preparation, please refer to Section 4.3 in this [paper](https://arxiv.org/pdf/2206.02146).


### Acknowledgments

This codebase borrows from the following [BasicSR](https://github.com/xinntao/BasicSR) and [ShiftNet](https://github.com/dasongli1/Shift-Net) repositories.

### Citation

If you find our work useful, please consider citing our paper in your research.


```
@article{ghasemabadi2024learning,
  title={Learning truncated causal history model for video restoration},
  author={Ghasemabadi, Amirhosein and Janjua, Muhammad K and Salameh, Mohammad and Niu, Di},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={27584--27615},
  year={2024}
}
```
