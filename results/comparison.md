# Method comparison (held_out, n=81)

| Method | Trainable parameters | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Cascade (geometric, CNN fallback) | 4,053,416 | 96.30% | 0.963 |
| Geometric head-yaw (Face Mesh) | 2 | 95.06% | 0.951 |
| EfficientNetB0 (fine-tuned) | 4,053,414 | 80.25% | 0.794 |
| Geometric iris offset (Face Mesh) | 2 | 62.96% | 0.621 |
| EfficientNetB0 (LFR pre-train only) | 4,053,414 | 59.26% | 0.549 |
| Majority-class baseline | 0 | 35.80% | 0.176 |
