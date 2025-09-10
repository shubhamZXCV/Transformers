# REPORT


### LOSS

#### Training Loss Per Batch
![train_batch_loss](plots/train_batch_loss.png)

#### Training Loss Per Epoch
![train_batch_loss](plots/train_epoch_loss.png)

#### Validation Loss Per Epoch
![train_batch_loss](plots/val_epoch_loss.png)

> From these plots we can conclude that ROPE converges slightly faster than Relative_Bias Positional encoding

### Translation Accuracy

#### BLEU SCORE

|             |Greedy|Beam|TOP-K|
|-------------|------|----|-----|
|**ROPE**         | 4.92 |4.61|3.08 |
|**RELATIVE BIAS**| 4.67 |4.78|3.18 |


