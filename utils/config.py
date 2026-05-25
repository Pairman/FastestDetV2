import os.path
from pathlib import Path
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

        dataset = data["DATASET"]
        self.train_txt: str = dataset["TRAIN"]
        self.val_txt: str = dataset["VAL"]
        self.names: list[str] = dataset["NAMES"]

        model = data["MODEL"]
        self.num_classes: int = int(model["NUM_CLASSES"])
        self.backbone_blocks: list[int] = [int(v) for v in model["BACKBONE_BLOCKS"]]
        self.backbone_channels: list[int] = [int(v) for v in model["BACKBONE_CHANNELS"]]
        self.backbone_name: str = model["BACKBONE_NAME"]
        self.input_size: list[int] = [int(v) for v in model["INPUT_SIZE"]]

        train = data["TRAIN"]
        self.learning_rate: float = float(train["LEARNING_RATE"])
        self.gamma: float = float(train["GAMMA"])
        self.warmup_epoch: int = int(train["WARMUP_EPOCH"])
        self.weight_decay: float = float(train["WEIGHT_DECAY"])
        self.momentum: float = float(train["MOMENTUM"])
        self.ema_decay: float = float(train["EMA_DECAY"])
        self.batch_size: int = int(train["BATCH_SIZE"])
        self.end_epoch: int = int(train["END_EPOCH"])
        self.milestones: list[int] = [int(v) for v in train["MILESTONES"]]

        augment = data["AUGMENT"]
        self.hsv_gain: list[float] = [float(v) for v in augment["HSV_GAIN"]]
        self.flip_p: float = float(augment["FLIP_P"])
        self.crop_min: float = float(augment["CROP_MIN"])
        self.narrow_max: float = float(augment["NARROW_MAX"])

        print(f"Loaded configs {path}")

if __name__ == "__main__":
    from pathlib import Path
    cfg = Config(str(Path(__file__).parents[1]/"configs/coco.yaml"))
    print(cfg.__dict__)
