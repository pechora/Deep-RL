# Deep Reinforcement Learning with Policy Gradients (Pong)

A simpler implementation of Deepmind's paper on [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) using atari-pong as the training environment.

###Objective:
We receive an image frame (a 210x160x3 byte array (integers from 0 to 255 giving pixel values)) and we get to decide if we want to move the paddle UP or DOWN (i.e. a binary choice). After every single choice the game simulator executes the action and gives us a reward: Either a +1 reward if the ball went past the opponent, a -1 reward if we missed the ball, or 0 otherwise. And of course, our goal is to move the paddle so that we get lots of reward. Pong is just a fun toy test case, something we play with while we figure out how to write very general AI systems that can one day do arbitrary useful tasks.

## References
[Deepmind](http://www.deepmind.com)
[Andrej Karpathy blog](http://karpathy.github.io/2016/05/31/rl/)
[Outlace](http://outlace.com/rlpart1.html)
