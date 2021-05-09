# Monte Carlo tree search hex solver using Deep Reinforcement Learning

This is the second of three projects in the AI programming course at NTNU. The group built an Actor Reinforcement Learner and has applied it to different instances of the game Hex. The Actor is a deep neural network that is trained on the probabilities given by the MCTS.

Figure 1 provides a high-level view of the actor.

<img src=images/actornetwork.png width="500" height="500">

**File structure:**

- agent
  - actor.py - the neural network actor
- environment
  - hexboard.py - the logic of the board
  - game_manager.py - updates the state of the game
- simulator
  - mcts.py - gives the probability distribution of a state, which is equal to the softmax of total counts of visits to all child nodes of that state
  - tree_node.py - node in the MCTS
    -tournament - tournament.py - used for playing models that are trained at different levels against each other

The config folder consists of different configs that have been used for the different instances of the game. In main.py it reads in these configs
and starts the whole training loop. The oht-folder is used for playing on the server created by the course administrators to test our models against their models. The nim-folder was used under developing the MCTS, to test that it worked on a less complicated game. Profiling was used to track the runtime of each element of the project.

|                Progression of Learning                 |                                Visualization of Game Play                                 |
| :----------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| <img src=images/learning.png width="475" height="350"> | ![Visualization of game play](https://media.giphy.com/media/4DNHzulV7JuVCIfnNr/giphy.gif) |
