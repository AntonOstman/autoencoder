
# Autoencoder test

Playing around with autoencoders to see which gives the best representation

For this simple test case the linear model seems to perform visually best.

## Models tested 

Linear autoencoder

Non linear autoencoder with relu on encoder and sigmoid on decoder

Non linear autoencoder with relu on encode and decode with various hidden layers


## Results

Upsampled encoded space, reconstruced image and original image

<div>
  <p>Linear model vs nonlinear vs nonlinearv2</p>
  <div>
    <td><img src="results/models/lr0.001linear/model4900.pth0.png" height="500"/></td>
    <td><img src="results/models/lr0.001nonlinear/model4900.pth0.png" height="500"/></td>
    <td><img src="results/models/lr0.001nonlinearv2/model4900.pth0.png" height="500"/></td>
  </div>

  <div>
    <td><img src="results/models/lr0.001linear/validation_loss_curve.png" width="166"/></td>
    <td><img src="results/models/lr0.001nonlinear/validation_loss_curve.png" width="166"/></td>
    <td><img src="results/models/lr0.001nonlinearv2/validation_loss_curve.png" width="166"/></td>
  </div>

  <p>The linear and first non linear remains stable, the nonlinearv2 starts overfitting</p>
</div>

# Dataset:

[Cats and Dogs Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset/data)

# Installation

Create a venv and install torch, opencv, numpy

Download the dataset and populate the project folders
