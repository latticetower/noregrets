"""This code is based on ranking loss implementation provided at
https://github.com/jiaxianyan/MBP. 
If you use it, please cite the original authors (or check for the additional citations in the repo above):

```{bibtex}
@article{Yan2023MultitaskBP,
  title={Multi-task bioassay pre-training for protein-ligand binding affinity prediction},
  author={Jiaxian Yan and Zhaofeng Ye and Ziyi Yang and Chengqiang Lu and Shengyu Zhang and Qi Liu and Jiezhong Qiu},
  journal={Briefings in Bioinformatics},
  year={2023},
  volume={25}
}
```
"""

import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    def __init__(self, embedding_out_dim: int, dropout: float=0.3, ntasks: int=2):
        super(RankingLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss() 
        self.embedding_out_dim = embedding_out_dim
        
        dim_list = [
            embedding_out_dim * 2,
            embedding_out_dim,
            embedding_out_dim // 2
        ]
        self.relation_mlp = nn.Sequential()
        for dim0, dim1 in zip(dim_list, dim_list[1:]):
            self.relation_mlp.extend([
                nn.Linear(dim0, dim1),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.BatchNorm1d(dim1)
            ])
        self.relation_mlp.append(nn.Linear(dim1, ntasks))

    @torch.no_grad()
    def get_rank_relation(self, y_A, y_B):
        target_relation = torch.zeros(y_A.size(), dtype=torch.long, device=y_A.device)
        target_relation[(y_A - y_B) > 0.0] = 1

        return target_relation.squeeze()

    def forward(self, output_embedding: torch.Tensor, target: torch.Tensor):
        batch_repeat_num = len(output_embedding)
        shift = max(batch_repeat_num // 2, 1)
        x_A, y_A = output_embedding, target
        x_B = torch.roll(output_embedding, shift, 0)
        y_B = torch.roll(target, shift, 0)

        relation = self.get_rank_relation(y_A, y_B)
        relation_pred = self.relation_mlp(torch.cat([x_A, x_B], dim=1))

        ranking_loss = self.loss_fn(relation_pred, relation)

        _, y_pred = nn.Softmax(dim=1)(relation_pred).max(dim=1)

        return ranking_loss, relation.squeeze(), y_pred