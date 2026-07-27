## Building and Training Deep Neural Networks.md

### CNNs - Part 1: Filters, Patterns, and Feature Maps
- Convolutional Filters: ustalizes the grayscale value of each pixel and its immediate neighbors (3x3 grid) to multiply with a grid of filter values also 3 by 3 that are all added together with a bias: creating a new grayscale value for that pixel
  - sliding that filter over an image is called Convolution
  - by doing this, you highight different features of the image
  - Ex: <img width="219" height="208" alt="image" src="https://github.com/user-attachments/assets/66842f51-095d-4f9c-82aa-03d3c2777bfe" />
 <img width="140" height="133" alt="image" src="https://github.com/user-attachments/assets/db835eba-d8e5-4e13-9cce-f86d095ac6fc" /> <img width="219" height="208" alt="image" src="https://github.com/user-attachments/assets/44320d0e-fc8f-4f70-8630-13c29dbccb05" />

    By using the filter with the grid values of those numbers, if the left and right of the pixel differ greatly, then the output with the filter will show great contrast. While if the grayscale value of the pixel's left and right are similar, then with the filter, the sum will be close to one as the left and right will cancel out, making the output darker

  - using these filters can help us reveal patterns that differentiate one image from another
- Convolutional Neural networks and find which filters will work the best and tune them to find specific patterns
- Create CNNs in PyTorch
<img width="746" height="157" alt="image" src="https://github.com/user-attachments/assets/824f49e8-786b-4de5-b82f-c37591fd871b" />

  - nn.Conv2d is used to define the CNN which represent that its a 2d filter for a 2d image
  - in_channels: number of color channels in each image (rbg has 3)
  - out_channels: how many filters the CNN uses, with each filter defining a feature; use multiple filters to capture multiple features in the image
  - kernel_size: the size of each filter (3 by 3 is common because it includes he picel and its nearest neighbors)
  - stride: how far each filter moves with each step (in pixels)
  - padding: for egdes and corners where a 3 by 3 cannot fit, the padding adds an imagenary edge around the edge set to zero by default 

### CNNs - Part 2: The Full Architecture
<img width="857" height="396" alt="image" src="https://github.com/user-attachments/assets/38eae4c3-baca-4339-b524-93aab51633fd" />

- The full model starts with two convulutional layers and ends in a fully connected layer (fully connected: every neuron in the input is connected to every neuron in the output
- start with a convolutional layer of one channel (grayscale), learn 32 different filters, with the size of the filter being 3 by 3 and a padding of 1
  - output: 32 distinct looking images that are the same images with different filters other know as feature maps or activation maps
- ReLU sets any negative value in the feature map to zero
  - helps learn more complexed image patterns
- Maxpool2d: pooling is a common tehnique used in CNNs to reduce the size of feature maps
  - takes the kernal size and divids the feature grid into equal sized kernel sized grids. In those grids the highest value is taken and formed into a kernel sized grid
  - this helps reduce data size and lets the model run faster and more smoothly
- __init__: if a class is a blueprint, then the __init__ fuction is where you assign specific details to objects you build
  - self: is always the first parameter of __init__. allows python to assign values to that object. Creates a permanet value attached to that object
  - Ex: self.name = name; name = name wouldn't work as the latter creates a local variable
- Fully connected layer: in this case is (64 * 7 * 7, 10) because it's 64 outputs in the previous layer with a maxpool layer of 7x7; 10 is because the final output is 10 (I think)
<img width="501" height="414" alt="image" src="https://github.com/user-attachments/assets/4e4d56df-7087-40ad-819e-a5fd70183267" />

- pass data through each of the convolutional layers
- flattens the layer before passing the data throught the final layer
- the final layer being the fully connected layer makes an prediction

### Train a CNN for Image Classification

 
