# Quixo
This Folder contains my solution for the implementation of an agent capable of winning a match of Quixo


## Players
The `players.py` file contains all of the available agents' classes:


### RandomPlayer()
The basic random agent initially provided.


### ManualPlayer()
A class implemented to allow any user to manually play the game step by step.\
When this agent is selected, the following instructions are provided on the terminal at the beginning of the game on how to provide the correct input in order for the class to work:\
`Select your moves based on 3 integers: X Y Shift`\
`X and Y are your coordinates. They range from 0 to 4 and go respectively from left to right (X) and from top to bottom (Y)`\
`Shift is the movement of an entire line of cells and can be:\n0: From Top    1: From Bottom    2: FromLeft    3: From Right`\
`Input Example: 4 1 2`\
`This means taking the piece from coordinates (4, 1) and shift the row from left to right`\


### QLPlayer()
An implementation of the Q-Learning algorithm used to make an agent capable of playing the game.\
Because of the sheer amount of possible states, it is unadvised to use this agent because of the massive amount of training required.\
After a training consisting of 100_000 games the dictionary was still not developed enough to provide the agent enough information to consistently win games. in fact, because of this issue, the agent mostly performs random moves at thest time, since a lot of the possible intermidiate states are missing and most of the present ones have still a quality value of 0, meaning that they have been explored only once.


### MinMaxPlayer()
An agent built on the principles of the MinMax strategy.\
This is the best performing agent implemented. It is capable of consistently winning games against the `RandomPlayer` and either win or prevent losses against the `ManualPlayer`.\
Notice: at the beginning of the file a constant `DEPTH_LEVELS` is declared. Its function is to provide the agent with the depth of the search tree. A minimum of 2 layers is recommended and is enough to achieve the aforementioned results. If the constant is set to 3, the program starts slowing down an might require some time to perform matches depending on the device currently running the program.


## Game
The `game.py` file contains the code that allows for the game to be run.\
Here the print function has been modified in order to print a more intuitive board
Moreover, other than the initially provided class `Game` (and corresponding methods) a new subclass has been added called `MyGame` in which some new methods have been provided.


### reset_board()
Function used to reset the board to its initial state.


### get_available_actions()
Function used to generate a tuple object containing all the possible moves an agent can perform.


### check_action()
Function usually called inside the `get_available_actions` function in order to quickly check the eligibility of a move.


### move()
Function used to directly call the `__move` method of the class `Game`.


### q_play()
Variation of the `play` function capable of performing training for the `QL_Player`.


## Main
The main file contains the main function in which you can declaree the agents that will play and a `n_games` variable, used to perform multiple matches (for example if the objective is to test an agent against a random player on more than a single game).\
Two other functions are present:\
`ql_training` (which is called when the constant `QL` at the top of the code is set to `True`) that performs training for the `QLPlayer`.\
`print_instructions` quick function to print the needed instructions to play a game manually








