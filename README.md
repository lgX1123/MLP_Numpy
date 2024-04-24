# COMP5329
Pure numpy MLP for multi-classification.

| Modules                | Baseline          | Best Model        |
| ---------------------- | ----------------- | ----------------- |
| Batch size             | 128               | 1024              |
| Learning rate          | 0.1               | 0.01              |
| Scheduler              | CosineAnnealingLR | None              |
| Epoch                  | 100               | 200               |
| Pre-processing         | Standardization   | None              |
| Number of Hidden layer | 2                 | 2                 |
| Hidden units           | [64, 32]          | [256, 128]        |
| Activations            | [Relu, Relu]      | [Relu, Relu]      |
| Weight initialisation  | Kaiming           | Kaiming           |
| Weight decay           | 5e-4              | 5e-4              |
| Optimizer              | SGD with Momentum | SGD with Momentum |
| Momentum               | 0.9               | 0.9               |
| Batch Normalisation    | Yes               | Yes               |
| Dropout rate           | 0.1               | 0.3               |
| Accuracy               | 53.03%            | 58.71%            |
