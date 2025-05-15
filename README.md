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
```

## ⚙️ Installation & Usage

Follow the steps below to set up and run DETSLideNet, **both locally** (without Docker) and **in Docker**.

---

### A. Prerequisites

- Python 3.8 or higher  
- Git  
- (Optional, for Docker) Docker & docker-compose

---

### B. Clone the repository

```
git clone https://github.com/Guemann-ui/DETSlideNet.git
cd DETSlideNet
```

### C. Data Download & Organization
Download the dataset from (Zenodo)[https://zenodo.org/records/10294997].

Extract under data/:
```
data/
├── train/
│   ├── Moxitaidi/
│   │   ├── img/
│   │   └── label/
│   ├── TiburonPeninsula/
│   │   ├── img/
│   │   └── label/
│   └── Jiuzhaivalley/
│       ├── img/
│       └── label/
└── test/
    ├── Moxitaidi/
    │   ├── img/
    │   └── label/
    ├── TiburonPeninsula/
    │   ├── img/
    │   └── label/
    └── Jiuzhaivalley/
        ├── img/
        └── label/

```
### D. 1. Local Setup (Without Docker)

#### 1. Create & activate virtual env
```
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows PowerShell
```
#### 2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
#### 3. Train
```
python main.py \
  --mode train \
  --region Moxitaidi,TiburonPeninsula,Jiuzhaivalley \
  --epochs 300 \
  --batch-size 16 \
  --learning-rate 0.01 \
  --sat-weight 0.4 \
  --img_size 128 \
  --output-dir checkpoints/run1
```
#### 4. Evaluate
```
python main.py \
  --mode test \
  --region Moxitaidi,TiburonPeninsula,Jiuzhaivalley \
  --load checkpoints/model_best.pth \
  --img_size 128
```
### D. 2. Docker Setup

#### 1. Build Docker image
```
docker build -t detslidenet:latest .
```
#### 2. Train in container
PowerShell / Windows:
```
docker run --rm \
  --mount type=bind,source="$(pwd)/data",target=/app/data \
  --mount type=bind,source="$(pwd)/checkpoints",target=/app/checkpoints \
  detslidenet:latest \
  --mode train --epochs 100 --batch-size 4 --region Moxitaidi --img_size 128
```
Bash / Linux / macOS:
```
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  detslidenet:latest \
  python main.py --mode train --output-dir checkpoints/run1
```

#### 3. Evaluate in container
PowerShell / Windows:
```
docker run --rm `
  --mount type=bind,source="$($pwd.Path)\data\data",target=/app/data/data `
  detslidenet:latest `
  --mode test `
  --region Moxitaidi,TiburonPeninsula,Jiuzhaivalley `
  --load checkpoints/model_best.pth `
  --img_size 128
```
Bash / Linux / macOS:
```
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/results:/app/results \
  detslidenet:latest \
  python main.py --mode test --load checkpoints/run1/model_best.pth
```
#### 4. (Optional) Docker Compose
```
docker-compose up --build
```
### E. Quick Commands
```
python main.py -m train	Train with default settings
python main.py -m test -f <ckpt>	Evaluate saved checkpoint
docker run …	Train or test in Docker container
```
