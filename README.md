# CHAI - DriveLM

This code implements CHAI on [DriveLM](https://arxiv.org/pdf/2312.14150)

To run the code:
```
python main.py --vlm gpt
```
Use the flag `--testing` to test the attack, otherwise run in training mode. 
Use the flag `--testing_ds` to use the testing dataset, otherwise run over the training dataset.

The images for traning are in file `training.json`.