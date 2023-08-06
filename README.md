# session12
Session 12


Explanation
============
This Package trains a Deep Neural Network on CIFAR data using a Custom ResNet model utilizing Pytorch Lighning

    
models/resnet18.py
------------------

The Custom Resnet18 Script is defined here 

PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax

    
    
utils/
-----

accuracy_utils.py                : Utility to calculate accuracy of training and testing 
cyclic_lr_plot.png               : Utility to plot clclic LR Plot
cyclic_lr_util.py
gradcamkz_util.py                : Gradcam Utility
misclassified_image_utils.py     : Utility to find misclassified images 
plot_metrics_utils.py            : Plot the Metrics
train_test_utils.py              : Train Test Utility
    
     

Training Log
-------------


INFO:pytorch_lightning.utilities.rank_zero:GPU available: True (cuda), used: True
INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO:pytorch_lightning.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs

Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./cifar-10-python.tar.gz

100%|██████████| 170498071/170498071 [01:15<00:00, 2267427.29it/s]

Extracting ./cifar-10-python.tar.gz to .
Files already downloaded and verified

WARNING:pytorch_lightning.loggers.tensorboard:Missing logger folder: /content/lightning_logs
INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:pytorch_lightning.callbacks.model_summary:
  | Name     | Type               | Params
------------------------------------------------
0 | model    | CustomResNet       | 6.6 M 
1 | accuracy | MulticlassAccuracy | 0     
------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.293    Total estimated model params size (MB)

/content/CustomResNet.py:76: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(out)

Epoch 23: 100%
176/176 [00:20<00:00, 8.67it/s, v_num=0, val_loss=0.696, val_acc=0.774]

INFO:pytorch_lightning.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=24` reached.

