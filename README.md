# crawler_robot

The aim here is to create a learning algorithm for a robot so that it learns to move forward. As of now , the robot under 
consideration is just a chassis with a limb having 2 joints at the very front of it. It has no sensors other than the 2 angle 
sensors for its 2 joints. Further work will be done on much more complicated robot systems with multiple limbs and sensors.

The 2 servos for the 2 joints are controlled using a neural network which is trained using a Genetic Algorithm. The architecture is 
same as of that used in the repo "nn_XOR_GA". This is basically a kind of reinforcement learning problem .

The library "pygame" and solving the mechanics of the bot was necessary for simulating the bot. ( The simulation is not really a 
very good one) . Also note that in the simulation, I have kept the robot fixed and made the ground movable as it is easier to code. 

The microcontroller code and video of the actual bot shall be updated soon.
