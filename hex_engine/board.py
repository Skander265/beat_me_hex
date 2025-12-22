from .piece import Piece

class Board:
    def __init__(self, size: int, env: dict = None):
        self.size = size
        self.env = env if env is not None else {}

    def get_env(self):
        return self.env

    def get_dimensions(self):
        return (self.size, self.size)

    def get_neighbours(self, r: int, c: int) -> dict:
        """
        Returns neighbors in the specific format your Agent expects:
        { "direction_name": (TYPE_STRING, (r, c)) }
        """
        offsets = {
            "top_right": (-1, 1),
            "top_left":  (-1, 0),
            "bot_left":  (1, -1),
            "bot_right": (1, 0),
            "left":      (0, -1),
            "right":     (0, 1)
        }
        
        result = {}
        for name, (dr, dc) in offsets.items():
            nr, nc = r + dr, c + dc
            
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                result[name] = ("OUTSIDE", (nr, nc))
                continue

            piece = self.env.get((nr, nc))
            if piece:
                result[name] = (piece.get_type(), (nr, nc))
            else:
                result[name] = ("EMPTY", (nr, nc))
                
        return result