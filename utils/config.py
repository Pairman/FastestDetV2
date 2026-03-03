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

        self.num_classes: int = data["MODEL"]["NUM_CLASSES"]
        self.input_size: list[int] = data["MODEL"]["INPUT_SIZE"]

        self.learning_rate: float = data["TRAIN"]["LEARNING_RATE"]
        self.warmup_epoch: int = data["TRAIN"]["WARMUP_EPOCH"]
        self.batch_size: int = data["TRAIN"]["BATCH_SIZE"]
        self.end_epoch: int = data["TRAIN"]["END_EPOCH"]
        self.milestones: list[int] = data["TRAIN"]["MILESTIONES"]

        print(f"Loaded configs {path}")

if __name__ == "__main__":
    from pathlib import Path
    cfg = Config(str(Path(__file__).parents[1]/"configs/coco.yaml"))
    print(cfg.__dict__)
