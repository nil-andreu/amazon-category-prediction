import os
import re

import numpy as np

from shared.model.train import MlpModel


def get_model_best_valid_score(shared_path: str):
    full_path: str = shared_path + "/data/model_ckpt/"
    all_models = os.listdir(full_path)

    if not all_models:
        return "", np.inf

    best_model_idx, best_model_score = 0, np.inf

    pattern = r"epoch=epoch=\d+-validation_loss=validation_loss=([0-9.]+).ckpt"

    for idx, model in enumerate(all_models):
        match = re.search(pattern, model)
        if match:
            validation_loss = float(match.group(1))
            if validation_loss < best_model_score:
                best_model_score, best_model_idx = validation_loss, idx

    return full_path + all_models[best_model_idx], best_model_score


def load_mlp_model(shared_path: str):
    os.makedirs(shared_path + "/data/model_ckpt", exist_ok=True)

    mlp = MlpModel()
    full_path, _ = get_model_best_valid_score(shared_path)

    if not full_path:
        return mlp

    mlp.load_from_checkpoint(full_path)

    return mlp
