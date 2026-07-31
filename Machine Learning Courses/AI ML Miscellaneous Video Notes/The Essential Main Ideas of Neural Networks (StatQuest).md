## The Essential Main Ideas of Neural Networks
- https://www.youtube.com/watch?v=CqOfi41LfDw
- https://www.youtube.com/watch?v=IN2XmBhILt4
- https://www.youtube.com/watch?v=68BZ5f7P94E

This will hopefully help you better visualize what a neural network is actually doing

## 3Blue1Brown
https://www.youtube.com/watch?v=aircAruvnKk
- A simple neural network starts with an input, its "activation," which is stored within a neuron.
- In a classification neural network, the last layer usually represents the percentage in decimals of the possibility of that output being the right classification.
- The task of identifying numbers can be broken down into a separate task for each layer, then built up from each layer to the text
  - For example, the first layer represents the pixel input, and the next signals a line segment that the pixels have lit up, and the next can be a whole bunch of segments together that make up a number
  - Layer 1: input; layer 2: little edges; layer 3: larger edges ot loops that build up numbers; layer 4: identification of the numbers
- Each neuron in one layer has a connection to each of the neurons in the next layer, and each of the connections has a weight attached that tells the network how much the neurons in the next layer matter based on the input of the previous layer
  - The weights and bias change after each full epoch and are changed to better values that can better capture the patterns of the data
- The weights that connect to one neuron in the next layer cause the neuron to light up only when certain neurons in the previous layer have high values
  - ie., the weights determine which neurons have the greatest impact on the neuron that they are connected to.
  - a high input with a high positive weight connecting the input to that neuron will cause the values for that neuron to be higher
    - this can affect the activations of the neurons in the next layer, with negative weights pushing away from a neuron  
  - ReLU is a great classification activation function because it shuts off neural pathways that have low(negative) values.
  - When making classification networks, loss = nn.CrossEntropyLoss has softmax built in, converting the output into values between 0 and 1
  - The bias tells the network how high of a value the sum of each weighted neuron has to be before it can get meaningfully active
  - The entire neural network between layers can be seen as a matrix multiplication with matrix addition
  <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d01ca697-19a3-47dc-9946-ee50de6a2671" /> 
- A softmax function can be used to make predictions as a percentage from 0 to 1
  - It sets the output to e's power and magnifies winners, with higher values being enlarged more than lower values 
  - It then adds all the modified outputs and divides each of the modified values by the sum, creating a value from 0 to 1
<img width="756" height="788" alt="image" src="https://github.com/user-attachments/assets/dd54a25d-55bb-4851-9521-dbe520723cf4" />

- The nn.CrossEntropyLoss function then takes the negative natural log of the softmax function to calculate how wrong the softmax function is
  - Since the output of the softmax function is always between 0 and 1, and the natural log of a number in those intervals are always negative, we add the negative sign in front
  - The further the value is from 1, the higher the error value
  <img width="1576" height="792" alt="image" src="https://github.com/user-attachments/assets/8041e15b-0efb-45bb-9973-6e8db7f0b992" />

[https://gemini.google.com/app/da9a9c5903894000?is_sa=1&is_sa=1&android-min-version=301356232&ios-min-version=322.0&campaign_id=bkws&utm_source=sem&utm_medium=paid-media&utm_campaign=bkws&pt=9008&mt=8&ct=p-growth-sem-bkws&gclsrc=aw.ds&gad_source=1&gad_campaignid=20108148196&gbraid=0AAAAApk5BhlSAqgbbm_HjbvwN6jw9214a&gclid=Cj0KCQjws83OBhD4ARIsACblj19cefWhcxeLbFQ27ng4jnyCT2IA3gA8lnskuoOcBAlD3FshNGFbt1AaAhdrEALw_wcB](https://gemini.google.com/app/da9a9c5903894000)
