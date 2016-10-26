# crawler_robot

The aim here is to create a learning algorithm for a robot so that it learns to move forward. As of now , the robot under 
consideration is just a chassis with a limb having 2 joints at the very front of it. It has no sensors other than the 2 angle 
sensors for its 2 joints. Further work will be done on much more complicated robot systems with multiple limbs and sensors.

The 2 servos for the 2 joints are controlled using a neural network with feedback (much like Recurrent NN but not exactly) which is trained using a Genetic Algorithm. The architecture is 
same as of that used in the repo "nn_XOR_GA". This is basically a kind of reinforcement learning problem .

The library "pygame" and solving the mechanics of the bot was necessary for simulating the bot. ( The simulation is not really a 
very good one) . Also note that in the simulation, I have kept the robot fixed and made the ground movable as it is easier to code. 

When implemented physically, the robot needs to be tracked to measure the displacement. This is achieved using a camera. Green patches are stuck on the bot which is then detected using the camera.
There is another separate green patch which is kept fixed and the distance of robot is measured with reference to this patch. This method makes the distance measurement more immune to camera disturabances.  This measured displacement becomes the fitness value for the chromosome.

It should also be mentioned that Arduino has very low computational power. Hence it has been used only to control the servos. All other computations including that of the neural network are performed on the laptop.

# crawler robot using a formal RNN ( basically optimization of RNN using Genetic Algorithm )

The Neural Network controller in this version is a formal RNN with the possible states encoded using a one-hot vector. For example [1 0 0 0 0 0 0 0 0] means both "theta1" and "theta2" of the two arms decrease. The architecture is much similar to that of a language generation model using RNN. One can experiment using a LSTM or GRU. But since the most important reason for using those architectures is that vanilla RNN s suffer from Gradient explosion (or vanishing) and the experiment here uses Genetic Algorithms, I don't think using LSTM would be advantageous as it will unnecessarily increase the number of parameters. Using NSGA-II and NSGA-III to optimize RNN for multiple objectives is being studied.
