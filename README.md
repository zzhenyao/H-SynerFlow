# H-SynerFlow

Code for **H-SynerFlow: Learning Synergetic Spatio-Temporal Flows via Hypergraph Structures for Action Quality Assessment**.

## Environment

| Package | Version |
| --- | --- |
| PyTorch | 2.4.1 |
| TorchVision | 0.19.1 |
| NumPy | 1.24.4 |
| PyYAML | 6.0.2 |

Run from the repository root:

```bash
conda create -n H-SynerFlow python=3.8.20 pip=24.2 -y
conda activate H-SynerFlow
pip install -r requirements.txt
pip install --no-deps dhg==0.9.4
pip install -e ./torchlight
```

DHG is installed with `--no-deps` because its declared `torch<2.0` constraint conflicts with PyTorch 2.4.1. Its other dependencies are included in `requirements.txt`.

## Data

### RG-skeleton

[RG](https://github.com/qinghuannn/ACTION-NET) is a rhythmic gymnastics dataset for action quality assessment which only provides raw video without skeleton data. We extracted skeletons from each frame in RG by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). The extracted skeleton data can be directly downloaded from [OneDrive](https://1drv.ms/u/c/14ab28d7828c2c4c/IQDbjGJj41JkSp0UcN3lfQ6aAYf4sQ77UOA9njlVrHELM44?e=oHdILN) or [BaiduYun](https://pan.baidu.com/s/1ELsLgQ7wmFI7MmS3THeG1A?pwd=nn5i).

```text
/data/AQAdata/RG/RG_long/
|-- BALL/
|   |-- BALL_data_train.npy
|   |-- BALL_label_train.pkl
|   |-- BALL_data_test.npy
|   `-- BALL_label_test.pkl
|-- CLUB/
|   |-- CLUB_data_train.npy
|   |-- CLUB_label_train.pkl
|   |-- CLUB_data_test.npy
|   `-- CLUB_label_test.pkl
|-- HOOP/
|   |-- HOOP_data_train.npy
|   |-- HOOP_label_train.pkl
|   |-- HOOP_data_test.npy
|   `-- HOOP_label_test.pkl
`-- RIBBON/
    |-- RIBBON_data_train.npy
    |-- RIBBON_label_train.pkl
    |-- RIBBON_data_test.npy
    `-- RIBBON_label_test.pkl
```

### Fis-V-skeleton

[Fis-V](https://github.com/chmxu/MS_LSTM) is a figure skating dataset. We applied the same data processing strategy to extract skeleton data. The extracted data can be directly downloaded from [OneDrive](https://1drv.ms/u/c/14ab28d7828c2c4c/IQCxUOlAHQFTRb8w5u8xxmE-AdjIH7i8SuEHCDv6v5XTp-Y?e=4Oj2Kl) or [BaiduYun](https://pan.baidu.com/s/1EVGI3YuJ2xYq-_j6UcKnXQ?pwd=4cis).

```text
/data/AQAdata/Fis-V/Fis-V_4500/
|-- PCS/
|   |-- PCS_data_train.npy
|   |-- PCS_label_train.pkl
|   |-- PCS_data_test.npy
|   `-- PCS_label_test.pkl
`-- TES/
    |-- TES_data_train.npy
    |-- TES_label_train.pkl
    |-- TES_data_test.npy
    `-- TES_label_test.pkl
```

## Training

Example for Ball; replace `ball.sh` with `club.sh`, `hoop.sh`, `ribbon.sh`, `pcs.sh`, or `tes.sh` for other tasks.

```bash
bash scripts/ball.sh --phase train
```


## Evaluation

Checkpoints: [OneDrive](https://1drv.ms/f/c/14ab28d7828c2c4c/IgD8BsAyf3zATZxr6CE0mxwjAYq4udTFokVRdn86No9Nbe8?e=KGYsJa) | [Baidu Netdisk](https://pan.baidu.com/s/1Ff-9ir-RpFr0UlBODIjvog?pwd=fip5).

Always pass a checkpoint trained for the selected task via `--weight_path`. The backbone weights in `pre_trained/` are only for initialization.

```bash
bash scripts/ball.sh --phase test --weight_path /path/to/ball.pt
```

## Outputs

Training logs, copied configs, source snapshots, and checkpoints are saved under:

```text
outputs/<user>/<dataset>/net.clip_target.Model/expN/
```

Each run creates a new `expN` directory, with the best saved checkpoint at `weights/best_model.pt`.

## Acknowledgements

Our work benefited from several outstanding projects in related areas, including: [DHG (hypergraph)](https://github.com/iMoonLab/DeepHypergraph), [ST-GCN (action recognition)](https://github.com/yysijie/st-gcn/tree/master), and [MAGR (action quality assessment)](https://github.com/ZhouKanglei/MAGR_CAQA).

If you have any questions, you can contact 📧email: yaozhen625 at gmail.com.
