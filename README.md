# Installation

## Conda
based on nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

```bash
# Step 1: Install Required Packages
pip install torch nvtx datasets

# Step 2: Install NVIDIA Apex
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Step 3: Install ColossalAI
git clone https://github.com/jing-4369/ColossalAI.git
cd ColossalAI
BUILD_EXT=1 pip install .
pip install transformers --upgrade
cd ..

# Step 4: Clone Zen Repository and install cuda extensions
git clone https://github.com/zhuangwang93/ZEN.git
cd ZEN
cd extensions/cuda/
python3 setup.py install
cd ../..

# Step 5: update hosts in hostfile

# Step 6: Run Training Script
bash run_all.sh
```


## Support Gemma for TP

Need add gemma model in ColossalAI/colossalai/shardformer to enable gemma.

## Nsys Profile
first install nsys
```
apt install nsight-systems-2024.5.1
```

add nsys profile before command
```bash
nsys profile -o test.nsys-rep --force-overwrite true colossalai run --nproc_per_node 4 train.py ...
```