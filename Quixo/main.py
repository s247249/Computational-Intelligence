import random
import numpy as np
from game import Game, MyGame, Move, Player
from players import RandomPlayer,  ManualPlayer, QLPlayer, MinMaxPlayer
from tqdm.auto import tqdm

QL = False


def ql_training():
    """
    Perform training if the QL_Player player has been selected
    """
    training_steps = 10_000
    
    player1 = QLPlayer()
    player2 = RandomPlayer()

    if type(player1) is type(QLPlayer()):
        ql_sign = "✖️ "
    elif type(player2) is type(QLPlayer()):
        ql_sign = "⭕"
    else:
        ql_sign = None
    if ql_sign:
        print(f"\nMinMax Player's sign = {minmax_sign}")

    min_exp_rate = 0.1
    max_exp_rate = 0.9
    exploration_rate_list = np.arange(min_exp_rate, max_exp_rate, max_exp_rate-min_exp_rate/training_steps).tolist()[::-1]
    player1.set_exploration_rate_list(exploration_rate_list)
    
    ql_players = list()
    ql_players.append(0)

    for steps in range(training_steps):
        player1.update_exploration_rate(steps)
        winner = g.q_play(player1, player2, ql_players)
        if winner == 0: win_counter += 1
        elif winner == 1: lose_counter += 1
    
    player1.set_exploration_rate(0)
    return player1, player2
    
def print_instructions():
    print("Select your moves based on 3 integers: X Y Shift")
    print("X and Y are your coordinates. They range from 0 to 4 and go respectively from left to right (X) and from top to bottom (Y)")
    print("Shift is the movement of an entire line of cells and can be:\n0: From Top    1: From Bottom    2: FromLeft    3: From Right")
    print("Input Example: 4 1 2\nThis means taking the piece from coordinates (4, 1) and shift the row from left to right")

if __name__ == '__main__':
    g = MyGame()
    n_games = 20
    
    if QL:
        player1, player2 = ql_training()
    
    else:
        player1 = MinMaxPlayer()
        player2 = RandomPlayer()

    player1_win_counter = 0
    player2_win_counter = 0

    if type(player1) is type(MinMaxPlayer()):
        minmax_sign = "✖️ "
    elif type(player2) is type(MinMaxPlayer()):
        minmax_sign = "⭕"
    else:
        minmax_sign = None
    
    if minmax_sign:
        print(f"\nMinMax Player's sign = {minmax_sign}")
    
    if type(player1) is type(ManualPlayer()) or type(player2) is type(ManualPlayer()):
        print_instructions()

    for steps in range(n_games):
        g.reset_board()
        winner = g.play(player1, player2)
        if winner == 0: player1_win_counter += 1
        elif winner == 1: player2_win_counter += 1
        print(f"\nFinished game #{steps+1}\tWinner: {winner}")
        g.print()

    print(f"\n Final results:\nPlayer ✖️   wins: {player1_win_counter}\nPlayer ⭕  wins: {player2_win_counter}")
    # print(f"Winner: Player {winner}")
