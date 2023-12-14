This project, named AutoVQE, is a Python-based application designed for video quality enhancement. It utilizes two main libraries, automm and autovqe, which need to be installed for the project to function properly.

Include the official implementation of "Video Compression Artifact Reduction by Fusing Motion Compensation and Global Context in a Swin-CNN based Parallel Architecture
" (AAAI'23)

## Getting started
```python
#install automm
python -m pip install -e /path/of/automm

#install auvqe
python -m pip install -e /path/of/autovqe
```
#### Data prepared
MFQEv2 dataset:
test frame:all frames: 7980
train frame:all frames: 38166
```
python AutoVQE/other/mfqe_data_prep/split_mfqev2_frames.py
```

### train
```bash
bash /DATA/jupyter/personal/AutoVQE/run.sh
```

### todo list
1. - [x] Release code
2. - [ ] Detailed ReadME
3. - [ ] Release model weights


https://github.com/WilliammmZ/AutoVQE/assets/18674350/c6a78470-f162-454f-9c02-a6cd7b80f57e


