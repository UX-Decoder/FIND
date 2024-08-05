# 🔍 FIND: Interfacing Foundation Models' Embeddings
:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2312.07532.pdf)\] &nbsp; :apple: \[[Try our Demo](http://find.xyzou.net:6789)\] &nbsp; :orange: \[[Walk through Project Page](https://x-decoder-vl.github.io/)\]

We introduce **FIND** that can **IN**terfacing **F**oundation models' embe**DD**ings in an interleaved shared embedding space. Below is a brief introduction to the generic and interleave tasks we can do!

by [Xueyan Zou](https://maureenzou.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Jianfeng Wang](http://jianfengwang.me/), [Jianwei Yang](https://jwyang.github.io/), [Mingyu Ding](https://dingmyu.github.io/), [Junyi Wei](https://scholar.google.com/citations?user=Kb1GL40AAAAJ&hl=en), [Zhengyuan Yang](https://zyang-ur.github.io/), [Feng Li](https://fengli-ust.github.io/), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en), [Shilong Liu](https://lsl.zone/), [Arul Aravinthan](https://www.linkedin.com/in/arul-aravinthan-414509218/), [Yong Jae Lee*](https://pages.cs.wisc.edu/~yongjaelee/), [Lijuan Wang*](https://scholar.google.com/citations?user=cDcWXuIAAAAJ&hl=zh-CN), 

** Equal Advising **

![FIND design](assets/images/teaser.jpg?raw=true)

## :rocket: Updates
* **[2023.12.3]**  We have a poster session @ NeurIPS24 for [SEEM](https://arxiv.org/pdf/2304.06718.pdf), feel free to visit us during 5:00-7:00pm (CT)!
* **[2023.12.2]**  We have released all the training, evaluation, and demo code!

## :bookmark_tabs: Catalog
- [x] Demo Code
- [x] Model Checkpoint
- [x] Comprehensive User Guide
- [x] Dataset
- [x] Training Code
- [x] Evaluation Code

## :hammer: Getting Started

<details open>
<summary>Install Conda</summary>
<pre>
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init
conda init zsh
</pre>
</details>

**Build Environment**
```
conda create --name find python=3.10
conda activate find
conda install -c conda-forge mpi4py
conda install -c conda-forge cudatoolkit=11.7
conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt
cd modeling/vision/encoder/ops
sh make.sh
cd ../../..
```

**Build Dataset**
<details open>
<summary>Data Structure</summary>
<pre>
data/
└── coco/
    ├── annotations/
    │   ├── entity_train2017.json
    │   ├── entity_val2017.json
    │   └── entity_val2017_long.json
    ├── panoptic_semseg_train2017/
    ├── panoptic_semseg_val2017/
    ├── panoptic_train2017/
    ├── panoptic_val2017/
    ├── train2017/
    └── val2017/
</pre>
</details>

**Run Demo**

**Run Evaluation**

**Run Training**

## 🗂️ Dataset
Explore through [🤗 Hugging Face: FIND-Bench](https://huggingface.co/datasets/xueyanz/FIND-Bench).

**Download Raw File:**
| entity_train2017.json | entity_val2017.json | entity_val2017_long.json |
|-----------------------|---------------------|--------------------------|
| [download](https://huggingface.co/datasets/xueyanz/FIND-Bench/resolve/main/entity_train2017.json)              | [download](https://huggingface.co/datasets/xueyanz/FIND-Bench/resolve/main/entity_val2017.json)            | [download](https://huggingface.co/datasets/xueyanz/FIND-Bench/resolve/main/entity_val2017_long.json)                 |


## ⛳ Checkpoint
|                   |          | COCO-Entity |      |      |       | COCO-Entity-Long |      |      |       |
|-------------------|----------|-------------|------|------|-------|------------------|------|------|-------|
|                   |          | cIoU        | AP50 | IR@5 | IR@10 | cIoU             | AP50 | IR@5 | IR@10 |
| ImageBIND (H)     | -        | -           | -    | 51.4 | 61.3  | -                | -    | 58.7 | 68.9  |
| Grounding-SAM (H) | -        | 58.9        | 63.2 | -    | -     | 56.1             | 62.5 | -    | -     |
| Focal-T           | [ckpt](https://huggingface.co/xueyanz/FIND/resolve/main/find_focalt_llama_x640.pt) | 74.9        | 79.5 | 43.5 | 57.1  | 73.2             | 77.7 | 49.4 | 63.9  |
| Focal-L           | [ckpt](https://huggingface.co/xueyanz/FIND/resolve/main/find_focall_llama_x640.pt) |             |      |      |       |                  |      |      |       |

## 🧑‍💻 Demo
* **Example Output**

<img width="400" alt="Screenshot 2023-12-13 at 10 28 05 AM" src="https://github.com/UX-Decoder/FIND/assets/11957155/48d84fb9-160c-4113-b50b-e7872dcde544">
<img width="400" alt="Screenshot 2023-12-13 at 10 31 36 AM" src="https://github.com/UX-Decoder/FIND/assets/11957155/b63582b2-45ca-4b3d-afd1-419770af2e2a">

## :framed_picture: FIND-Bench Visualization
<img width="400" alt="Screenshot 2024-08-05 at 3 50 54 PM" src="https://github.com/user-attachments/assets/541d5761-88f9-4797-ba07-66effcdd3e45">
<img width="400" alt="Screenshot 2024-08-05 at 3 50 46 PM" src="https://github.com/user-attachments/assets/dfece581-578a-4b41-9c18-d957f5868dcb">


## 📚 Reference

