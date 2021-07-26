# Slot Tagging
Practice slog tagging using CNN and LSTM

## Download models and data
```shell
# model will be downloaded as ckpt/slot/best.pt
bash download.sh
```

## Slot Tagging Training
```shell
python train_slot.py
```

## Slot Tagging Prediction
```shell
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```