from .board import Board
from .piece import Piece
from .action import Action
from .player import Player

class GameState:
    def __init__(self, board: Board, next_player_type: str):
        self.board = board
        self.next_player_type = next_player_type

    def get_rep(self):
        return self.board

    def get_next_player(self):
        return Player(self.next_player_type)

    def generate_possible_light_actions(self):
        size = self.board.size
        env = self.board.get_env()
        for r in range(size):
            for c in range(size):
                if (r, c) not in env:
                    yield Action((r, c), self.next_player_type)

    def apply_action(self, action: Action):
        pos = action.data["position"]
        
        new_env = self.board.get_env().copy()
        new_env[pos] = Piece(self.next_player_type)
        
        next_p = "R" if self.next_player_type == "B" else "B"
        
        return GameState(Board(self.board.size, new_env), next_p)

    def is_done(self):
        return self.check_win("B") or self.check_win("R")

    def get_scores(self):
        return {
            "B": 1.0 if self.check_win("B") else 0.0,
            "R": 1.0 if self.check_win("R") else 0.0
        }

    def check_win(self, p_type: str) -> bool:
        size = self.board.size
        env = self.board.get_env()
        
        starts = []
        if p_type == "B": # Left -> Right
            target_check = lambda c: c == size - 1
            for r in range(size):
                if env.get((r, 0)) and env[(r, 0)].get_type() == "B":
                    starts.append((r, 0))
        else: # Top -> Bottom
            target_check = lambda r: r == size - 1
            for c in range(size):
                if env.get((0, c)) and env[(0, c)].get_type() == "R":
                    starts.append((0, c))

        if not starts:
            return False

        # BFS 
        queue = list(starts)
        visited = set(starts)

        while queue:
            curr_r, curr_c = queue.pop(0)
            
            if (p_type == "B" and target_check(curr_c)) or \
               (p_type == "R" and target_check(curr_r)):
                return True

            neighbors = self.board.get_neighbours(curr_r, curr_c)
            
            for _, (n_type, n_pos) in neighbors.items():
                if n_type == p_type and n_pos not in visited:
                    visited.add(n_pos)
                    queue.append(n_pos)
                    
        return False