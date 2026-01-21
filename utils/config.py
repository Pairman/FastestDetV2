import yaml

class Config:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.train_txt: str = data["DATASET"]["TRAIN"]
        self.val_txt: str = data["DATASET"]["VAL"]
        self.names: str = data["DATASET"]["NAMES"]

        self.num_classes: int = data["MODEL"]["NUM_CLASSES"]
        self.input_width: int = data["MODEL"]["INPUT_WIDTH"]
        self.input_height: int = data["MODEL"]["INPUT_HEIGHT"]

        self.learning_rate: float = data["TRAIN"]["LEARNING_RATE"]
        self.warmup_epoch: int = data["TRAIN"]["WARMUP_EPOCH"]
        self.batch_size: int = data["TRAIN"]["BATCH_SIZE"]
        self.end_epoch: int = data["TRAIN"]["END_EPOCH"]
        self.milestones: list[int] = data["TRAIN"]["MILESTIONES"]

        print(f"Loaded configs {path}")
