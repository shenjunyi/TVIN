# Transfer Value Iteration Networks in Tensorflow
Framwork of Transfer Value Iteration Networks
![](https://github.com/shenjunyi/Transfer-VIN/blob/master/TVIN/tvin.jpg)
## Datasets
Generate the training data for target domain. Run the [Grid-world-Generator](https://github.com/shenjunyi/Grid-world-Generator) to get any grid-world training data you want, and training the different TVIN models.
 
Alternatively, you can use the existing data files in the data folder ```data/gridworld_16.mat```, ```data/gridworld_8.mat```.
## Requires
* Python >= 3.6
* TensorFlow >= 1.0
* SciPy >= 0.18.1 
* Numpy >= 1.12.1
## Training
### Pre-trained Model
Pre-trained 8\*8 Grid-world VIN model is saved in ```result``` directory, you can just restore the existing model in this folder.
### Target 16\*16 Grid-world domain
Train the TVIN for 16\*16 grid-world domain with parameters
```
python3 a8_train.py --lr 0.001 --epochs 30 --k 20 --batch_size 12
```
Monitor training progress in tensorboard by changing parameter ```config.log``` to ```True``` and launch ```tensorboard --logdir /tmp/vin/```. The log directory can be defined by yourself.<br>

#### Other parameters:

* ```lr```: Learning rate for RMSProp.
* ```epochs```: Maximum epochs to train for. Default: 30
* ```k```: Number of value iterations. Recommended: [10 for 8x8, 20 for 16x16]
* ```ch_i```: Channels in input layer. Default: 2, i.e. obstacles image and goal image.
* ```ch_h```: Channels in initial hidden layer (~ reward function)
* ```ch_q1```: Transfer channels in q layer (~ transfer actions) in TVIN-module.
* ```ch_q2```: New channels in q layer (~new actions) in TVIN-module. 
* ```batchsize```: Batch size. 
* ```statebatchsize```: Number of state inputs for each sample.

## Performance
![](https://github.com/shenjunyi/Transfer-VIN/blob/master/TVIN/ex2_2.jpg)
