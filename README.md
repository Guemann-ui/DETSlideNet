# DETSLideNet

> Dual-branch UAV+Satellite multi-scale segmentation network

**DETSlideNet** fuses high-resolution UAV imagery and wide-area satellite data via cross-attention and a flexible decoder, for accurate segmentation across scales.

---

## 🚀 Features

- **Dual-branch encoders** (SwinTransformer) for UAV & SAT streams  
- **Cross-attention fusion** at multiple scales  
- **Flexible decoder blocks** with optional skip connections  
- **Joint loss** combining UAV-supervision & SAT-supervision  
- End-to-end PyTorch implementation with training & evaluation scripts

---

## 📁 Repository Structure

```text
├── .github
│   └── workflows
│       └── ci.yml
├── checkpoints/           # saved model weights
├── data/                  # expected directory structure for datasets
│   ├── train/
│   └── test/
├── src/
│   ├── dataloader.py      # CASLDataset
│   ├── nets.py            # DETSLideNet and submodules
│   ├── train.py           # train_net
│   ├── test.py            # eval_net
│   └── utils/
│       ├── losses.py
│       └── module.py
├── tests/                 # unit & integration tests
│   ├── test_dataloader.py
│   ├── test_nets.py
│   └── test_train_eval.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── CODE_OF_CONDUCT.md
└── CONTRIBUTING.md
