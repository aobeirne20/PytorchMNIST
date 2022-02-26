# PytorchMNIST
Pytorch-based CNN to train on and classify the MNIST database. The number of layers in the network can be customized, but the input dimensions are limited by the traditional size of the MNIST data (28 x 28 pixels). 
As of now (Feb 2022), Pytorch is only compatible with Python 3.8

Includes GUI where the user can draw digits to test them out, and check the results of the network. Users can also see both the one-hot output and the predictive probabilities for each possible digit. Additional functionality also allows users to reset the GUI, save the current NN file as a .pth, and manually correct a misidentified digit.  
![CaptureGUI](https://user-images.githubusercontent.com/56138845/155859247-77b9c844-279d-4cca-98bb-79a16ecaad37.PNG)

Check CUDA usage before running and ensure compatibility with local hardware.
