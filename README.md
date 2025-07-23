# noregrets: No regrets, only losses

Here I want to store the collection of pytorch losses to reuse them in the future. Most probably I'll make them lightning-compatible. This is a collection of scripts

### Installation

```bash
git clone git@github.com:latticetower/noregrets.git
cd noregrets
pip install .
```

## Losses list

### `ranking_loss.RankingLoss`

The implementation is based on the ranking loss provided as a part of MBP framework.
If you use it, please cite the original authors (or check for the additional citations in their repository https://github.com/jiaxianyan/MBP):

```{bibtex}
@article{Yan2023MultitaskBP,
  title={Multi-task bioassay pre-training for protein-ligand binding affinity prediction},
  author={Jiaxian Yan and Zhaofeng Ye and Ziyi Yang and Chengqiang Lu and Shengyu Zhang and Qi Liu and Jiezhong Qiu},
  journal={Briefings in Bioinformatics},
  year={2023},
  volume={25}
}
```
