import torch
from fairseq import checkpoint_utils

from vencoder.encoder import SpeechEncoder


class ContentVec768L12(SpeechEncoder):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        super().__init__()
        print("load model(s) from {}".format(vec_path))
        self.hidden_dim = 768
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.model = models[0].to(self.dev)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.cpu(),  # 强制送到CPU
          "padding_mask": padding_mask.cpu(),
          "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.model.cpu().extract_features(**inputs)  # 保证模型在CPU
        return logits[0].transpose(1, 2).to(self.dev)  # 关键：转回主设备
