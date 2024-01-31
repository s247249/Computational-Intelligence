from game import Game, MyGame, Move, Player
import numpy as np
import random
import math

DEPTH_LEVELS = 4

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class ManualPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
    
    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        print("Current board:")
        game.print()
        available_actions = game.get_available_actions(game.get_current_player())
        action = -1
        input_piece = None
        while action not in available_actions:
            input_piece = input("Your Move: ")
            split_input = input_piece.split(' ')

            if len(split_input) != 3:
                print("Unacceptable input")
                continue
            
            # obtain the needed paramethers 
            from_pos = (int(split_input[0]), int(split_input[1]))
            input_move = int(split_input[2])

            # convert the move parameter to the correct type
            if input_move == 0:
                move = Move.TOP
            elif input_move == 1:
                move = Move.BOTTOM
            elif input_move == 2:
                move = Move.LEFT
            elif input_move == 3:
                move = Move.RIGHT
            else:
                move = -1
            action = (from_pos, move)

            if action not in available_actions:
                print("Move unavailable, try again\n")
        
        # generate a new game instance in order to print the board after the player's move has been performed
        game_instance = MyGame(game.get_board())
        game_instance.move(from_pos, move, game.get_current_player())
        print("Board after your move:")
        game_instance.print()
        
        return from_pos, move

class QLPlayer(Player):
    def __init__(self, learning_rate=0.3, exploration_rate=0.9, discount_factor=0.1):
        """
        Init of the PLayer

        Args:
            index: the index of the ql_player
            learning_rate: alpha
            exploration_rate: epsilon
            discount_factor: gamma
        """
        self.index = None
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.exploration_rate_list = list()

        self.available_actions = []
        board = self.game.get_board()
        self.q_table = {}
        # hashable lists used to update the q_table dictionary
        self.h_old_board = str(board)
        self.h_board = str(board)
        self.taken_action = None
    
    def set_exploration_rate(self, new_exploration_rate):
        self.exploration_rate = new_exploration_rate
    
    def set_exploration_rate_list(self, exp_rate_list):
        self.exploration_rate_list = exp_rate_list
    
    def get_q_table(self):
        return self.q_table

    def update_exploration_rate(self, step):
        """
        Update the exploration rate based on a previously provided list and the current training step

        Args:
            step: current training step
        
        Returns:
            the selected value if present in the table, 0 otherwise
        """
        self.set_exploration_rate(self.exploration_rate_list[step])

    def get_q_value(self, action):
        """
        Get a value from the quality table, given an action (the current state of the board gets handled separately)

        Args:
            action: the action to perform (from_pos, move)
        
        Returns:
            the selected value if present in the table, 0 otherwise
        """
        return self.q_table.get((self.h_board, action), 0.0)
    
    def update_q_table(self, reward):
        """
        Update values of the quality table (after the opponent has made their move)

        Args:
            reward: the obtained reward
        """
        # update the state of the current board and the list of available positions
        board = self.game.get_board()
        self.h_board = str(board)
        # self.check_available_pos()

        self.available_actions = self.game.get_available_actions(0)
        
        # make a list of all possible future actions (next action only, move unchecked)
        future_actions = (action for action in self.available_actions)
        # get all q_values from possible future actions 
        next_q_values = np.array([self.get_q_value(future_action) for future_action in future_actions])
        
        # math: Q(st, at) ← Q(st, at) * (1 − αt (st , at)) + αt(st , at) * [ Rt+1 + γ * max_Q (st+1, at+1) ]
        #           [ Rt+1 +          γ           * max_Q (st+1, at+1) ] 
        new_value = reward + self.discount_factor * (np.max(next_q_values) if len(next_q_values) > 0 else 0.0)
        # Q(st, at)                                         ←           Q(st, at)                  *  ( 1 − αt (st , at) )    +   αt(st , at)      * [ ... ]
        self.q_table[(self.h_old_board, self.taken_action)] =  self.get_q_value(self.taken_action) * (1 - self.learning_rate) + self.learning_rate * new_value

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        """
        Decide on a move to make

        Returns:
            from_pos, move: the position from which to take the piece from and the sliding to perform
        """
        self.game = game
        if self.index == None:
            self.index = self.game.get_current_player()
        # obtain current board and save it hashable format(str)
        board = self.game.get_board()
        self.h_board = str(board)
        self.h_old_board = str(board)
        # update the list of available positions
        # self.check_available_pos()

        self.available_actions = self.game.get_available_actions(0)

        # decide between exploration and exploitation
        explore = np.random.rand() < self.exploration_rate
        if explore:
            from_pos = random.choice(self.available_actions)[0]
            move = np.random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        else:
            # get values from the current state and possible_actions
            q_values = [self.get_q_value(action) for action in self.available_actions]
            # find the highest value 
            max_q_value = np.max(q_values)
            # take the action (from_pos, move) with the highest value
            # the "choice" favors randomness in case of multiple actions having the same q_value
            chosen_index = np.random.choice(np.where(q_values == max_q_value)[0])
            from_pos = self.available_actions[int(chosen_index / 4)][0]
            move = self.available_actions[chosen_index % 4][1]

        self.taken_action = (from_pos, move)
        
        return from_pos, move

class MinMaxPlayer(Player):
    def __init__(self) -> None:
        self.player_index = None
        self.opponent_index = None

    def check_close_win(self, board, player_index) -> int:
        """
        Check if a player is close to winning the current game instance

        Args:
            board: the state of the board in the current game instance
            player_index: the player index to check for close wins

        Returns:
            the current count of states where the provided player (index) is close to winning
        """
        cnt = 0

        for r in range(0, 5):
            row = board[r, :]
            col = board[:, r]
            y = np.where(row == player_index)[0]
            x = np.where(col == player_index)[0]
            if len(y) == 4:
                cnt += 1
            if len(x) == 4:
                cnt += 1
        
        inv_board = board[:, ::-1]
        diagonals = [np.diagonal(board), np.diagonal(inv_board)]
        d1 = np.where(diagonals[0] == player_index)[0]
        d2 = np.where(diagonals[1] == player_index)[0]
        if len(d1) == 4:
            cnt += 1
        if len(d2) == 4:
            cnt += 1

        return cnt
    
    def evaluate(self, game_instance: MyGame, player_index: int, depth: int) -> int:
        """
        Evaluate the current state of the board after reaching a terminal state (meaning a player has won or the provided tree depth has been reached)

        Args:
            game_instance: an object of class 'MyGame' used to check and perform moves without changing the current board
            player_index: the index of the player that would perform the next move (used to check if they're close to winning the game)
            depth: the remaining depth of the tree

        Returns:
            an integer value containing the final evaluation of the provided state of the board
        """
        winner = game_instance.check_winner()
        board = game_instance.get_board()

        if winner == self.player_index:
            return 100 + depth
        elif winner == self.opponent_index:
            return -(100 + depth)
        # no winner yet -> reached depth 0
        else:
            #check if the current player is close to winning
            close_win_value = self.check_close_win(board, player_index) * 10
            if(player_index == self.opponent_index):
                close_win_value *= -1
            
            # favor piece ownership
            uniques = np.unique(board, return_counts=True)
            unique_cnt = len(uniques[0])
            # if it contains blank pieces
            if unique_cnt == 3:  
                max_pieces = uniques[1][1]
                min_pieces = uniques[1][2]
            # if at lesat one action has been taken
            elif unique_cnt > 1:
                max_pieces = uniques[1][0]
                min_pieces = uniques[1][1]
            else:
                print("in else")
                max_pieces = 0
                min_pieces = 0
            
            return max_pieces - min_pieces + close_win_value

    def alpha_beta(self, game_instance: MyGame, alpha, beta, depth, p_index) -> tuple:
        """
        Follow the minmax algorithm with alphabeta pruning in order to chose the move to perform

        Args:
            alpha: the value of the max-player's move
            beta: the value of the min-player's move
            depth: the remaining depth of the tree (when 0 is reached, the algorithm stops)
            p_index: the index of the player that has to choose an action

        Returns:
            a value between alpha and beta (depending on the player's current turn) and the action associated to said value
        """
        winner = game_instance.check_winner()
        # return board value if we hit a final state
        if winner != -1 or depth == 0:
            score = self.evaluate(game_instance, p_index, depth)
            return (score, None)
        
        depth -= 1
        best_action = None
        # max_player
        if p_index == self.player_index:
            for action in game_instance.get_available_actions(self.player_index):
                # make a copy of the current game_state and play the move (iterating)
                new_game_instance = MyGame(game_instance.get_board())
                new_game_instance.move(action[0], action[1], self.player_index)
                # evaluate the move and perform alpha-beta pruning
                val = self.alpha_beta(new_game_instance, alpha, beta, depth, self.opponent_index)[0]
                if val > alpha:
                    alpha = val
                    best_action = action
                if alpha >= beta:
                    break
            # the max_player returns the value of alpha and the best action they can take
            return (alpha, best_action)
        # min_player
        else:
            for action in game_instance.get_available_actions(self.opponent_index):
                new_game_instance = MyGame(game_instance.get_board())
                new_game_instance.move(action[0], action[1], self.opponent_index)
                # evaluate the move and perform alpha-beta pruning
                val = self.alpha_beta(new_game_instance, alpha, beta, depth, self.player_index)[0]
                if val < beta:
                    beta = val
                    best_action = action
                if alpha >= beta:
                    break
            # the min_player returns the value of beta and the best action they can take
            return (beta, best_action)

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        self.game = game
        #if it's the first call, save the value of both player's indexes
        if self.player_index == None:
            self.player_index = self.game.get_current_player()
            self.opponent_index = (self.player_index + 1) % 2
        
        depth = DEPTH_LEVELS
        action = self.alpha_beta(self.game, -math.inf, math.inf, depth, self.player_index)[1]
        #      from_pos,  move
        return action[0], action[1]

