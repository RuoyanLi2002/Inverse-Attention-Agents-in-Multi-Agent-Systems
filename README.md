# Inverse-Attention-Agents-in-Multi-Agent-Systems

## Installation
To install the required Python packages, run:
```bash
pip install -r requirements.txt
```

## Training Instructions

### Training the Gradient Field
- To train the **agent gradient field**, run:
  ```bash
  python targfupdate/train_wolf.py
  ```
- To train the **wall gradient field**, run:
  ```bash
  python targfupdate/train_wall.py
  ```

### Training the Agent
To train the agent, run:
```bash
python mappo/onpolicy/scripts/train/train_mpe_scripts/train_mpe_tag.sh
```

### Cross Competition
To perform cross competition, run:
```bash
python mappo/onpolicy/scripts/cross_comp.sh
```

## Citation
If you use our work, please cite:

```bibtex
@inproceedings{
  long2025inverse,
  title={Inverse Attention Agent in Multi-Agent System},
  author={Qian Long and Ruoyan Li and Minglu Zhao and Tao Gao and Demetri Terzopoulos},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OaoDVZntGe}
}
```

