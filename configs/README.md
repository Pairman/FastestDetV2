## Configurations

The .yaml configurations file specifies dataset paths, model settings, and training hyperparameters. Dataset could be either in Darknet format or YOLO format. Class names can also be in a single text file with each line representing a class name.

```yaml
DATASET:
  # Path to training images list file (darknet style) or directory (yolo style)
  TRAIN: "/data/datasets/coco2017/images/train2017"
  # Path to evaluation images list file (darknet style) or directory (yolo style)
  VAL: "/data/datasets/coco2017/images/val2017"
  # Path to class names list file or list of class names
  NAMES: [person, ..., toothbrush]
MODEL:
  # Number of classes
  NUM_CLASSES: 80
  # Input width and height
  INPUT_SIZE: [352, 352]
TRAIN:
  # Initial learning rate
  LEARNING_RATE: 0.003
  # Gamma for learning rate decay
  GAMMA: 0.05
  # Number of warm-up epochs
  WARMUP_EPOCH: 5
  # Weight decay factor for non-BN parameters
  WEIGHT_DECAY: 1e-4
  # Momentum factor for SGD optimizer
  MOMENTUM: 0.949
  # EMA decay factor
  EMA_DECAY: 0.9998
  # Batch size
  BATCH_SIZE: 96
  # Total training epochs
  END_EPOCH: 300
  # Epochs for learning rate decay
  MILESTONES: [120, 270]
```