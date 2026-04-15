import os.path
import yaml

class Config:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        if isinstance(data["DATASET"]["NAMES"], str):
            if not os.path.exists(data["DATASET"]["NAMES"]):
                raise FileNotFoundError(data["DATASET"]["NAMES"])
            with open(data["DATASET"]["NAMES"], "r") as f:
                data["DATASET"]["NAMES"] = [l.strip() for l in f.readlines()]

        self.train_txt: str = data["DATASET"]["TRAIN"]
        self.val_txt: str = data["DATASET"]["VAL"]
        self.names: list[str] = data["DATASET"]["NAMES"]

        self.num_classes: int = int(data["MODEL"]["NUM_CLASSES"])
        self.input_size: list[int] = [int(v) for v in data["MODEL"]["INPUT_SIZE"]]

        self.learning_rate: float = float(data["TRAIN"]["LEARNING_RATE"])
        self.gamma: float = float(data["TRAIN"]["GAMMA"])
        self.warmup_epoch: int = int(data["TRAIN"]["WARMUP_EPOCH"])
        self.weight_decay: float = float(data["TRAIN"]["WEIGHT_DECAY"])
        self.momentum: float = float(data["TRAIN"]["MOMENTUM"])
        self.ema_decay: float = float(data["TRAIN"]["EMA_DECAY"])
        self.batch_size: int = int(data["TRAIN"]["BATCH_SIZE"])
        self.end_epoch: int = int(data["TRAIN"]["END_EPOCH"])
        self.milestones: list[int] = [int(v) for v in data["TRAIN"]["MILESTIONES"]]

        print(f"Loaded configs {path}")

if __name__ == "__main__":
    from pathlib import Path
    cfg = Config(str(Path(__file__).parents[1]/"configs/coco.yaml"))
    print(cfg.__dict__)
