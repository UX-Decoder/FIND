# 🔍 FIND: Interface Foundation Models' Embeddings
:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2312.07532.pdf)\] &nbsp; :apple: \[[Try our Demo](http://find.xyzou.net:6789)\] &nbsp; :orange: \[[Walk through Project Page](https://x-decoder-vl.github.io/)\]

We introduce **FIND** that can **IN**terfacing **F**oundation models' embe**DD**ings in an interleaved shared embedding space. Below is a brief introduction of all the generic and interleave tasks we can do!

<!-- by [Xueyan Zou*](https://maureenzou.github.io/), [Jianwei Yang*](https://jwyang.github.io/), [Hao Zhang*](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en),  [Feng Li*](https://fengli-ust.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Jianfeng Wang](http://jianfengwang.me/), [Lijuan Wang](https://scholar.google.com/citations?user=cDcWXuIAAAAJ&hl=zh-CN), [Jianfeng Gao^](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F), [Yong Jae Lee^](https://pages.cs.wisc.edu/~yongjaelee/), in **NeurIPS 2023**. -->

![FIND design](assets/images/teaser.png?raw=true)

## :rocket: Updates
* **[2023.12.3]**  We have a poster session@NeurIPS24 for [SEEM](https://arxiv.org/pdf/2304.06718.pdf), feel free to visit us during 5:00-7:00pm (CT)!
* **[2023.12.2]**  We have released all the training, evaluation, and demo code!

## :bookmark_tabs: Catalog
- [x] Demo Code
- [x] Model Checkpoint
- [x] Comprehensive User Guide
- [x] Dataset
- [x] Training Code
- [x] Evaluation Code

## :hammer: Getting Started
<details>
<summary><span style="font-weight: bold;">Installation</span></summary>
  #### llll
</details>
<br>
* Running demo from zero.
```sh
```

## :coconut: Dataset
| entity_train2017.json | entity_val2017.json | entity_val2017_long.json |
|-----------------------|---------------------|--------------------------|
| [download](https://huggingface.co/xueyanz/FIND/resolve/main/entity_train2017.json)              | [download](https://huggingface.co/xueyanz/FIND/resolve/main/entity_val2017.json)            | [download](https://huggingface.co/xueyanz/FIND/resolve/main/entity_val2017_long.json)                 |

## :kiwi_fruit: Checkpoint
|                   |          | COCO-Entity |      |      |       | COCO-Entity-Long |      |      |       |
|-------------------|----------|-------------|------|------|-------|------------------|------|------|-------|
|                   |          | cIoU        | AP50 | IR@5 | IR@10 | cIoU             | AP50 | IR@5 | IR@10 |
| ImageBIND (H)     | -        | -           | -    | 51.4 | 61.3  | -                | -    | 58.7 | 68.9  |
| Grounding-SAM (H) | -        | 58.9        | 63.2 | -    | -     | 56.1             | 62.5 | -    | -     |
| Focal-T           | [ckpt](https://huggingface.co/xueyanz/FIND/resolve/main/find_focalt_llama_x640.pt) | 74.9        | 79.5 | 43.5 | 57.1  | 73.2             | 77.7 | 49.4 | 63.9  |
| Focal-L           | [ckpt](https://huggingface.co/xueyanz/FIND/resolve/main/find_focall_llama_x640.pt) |             |      |      |       |                  |      |      |       |

## :mushroom: Demo
* **Example Output**

<img width="400" alt="Screenshot 2023-12-13 at 10 28 05 AM" src="https://github.com/UX-Decoder/FIND/assets/11957155/48d84fb9-160c-4113-b50b-e7872dcde544">
<img width="400" alt="Screenshot 2023-12-13 at 10 31 36 AM" src="https://github.com/UX-Decoder/FIND/assets/11957155/b63582b2-45ca-4b3d-afd1-419770af2e2a">

