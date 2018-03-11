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
## Traning
#### Pre-trained Model
Pre-trained 8\*8 Grid-world VIN model is saved in ```result``` directory, you can just restore the existing model in this folder.
#### Target 16\*16 Grid-world domain
Flags:

* datafile: The path to the data files.
* imsize: The size of input images. One of: [8, 16, 28]
* lr: Learning rate with RMSProp optimizer. Recommended: [0.01, 0.005, 0.002, 0.001]
* epochs: Number of epochs to train. Default: 30
* k: Number of Value Iterations. Recommended: [10 for 8x8, 20 for 16x16, 36 for 28x28]
* l_i: Number of channels in input layer. Default: 2, i.e. obstacles image and goal image.
* l_h: Number of channels in first convolutional layer. Default: 150, described in paper.
* l_q: Number of channels in q layer (~actions) in VI-module. Default: 10, described in paper.
* batch_size: Batch size. Default: 128

## Performance
