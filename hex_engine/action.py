class Action:
    """
    Represents a move in the game.
    """
    def __init__(self, position: tuple[int, int], piece_type: str = None):
        self.data = {
            "position": position,
            "piece": piece_type
        }

    def get_position(self):
        return self.data["position"]

    def __eq__(self, other):
        return self.data["position"] == other.data["position"]

    def __hash__(self):
        return hash(self.data["position"])

    def __str__(self):
        return f"Action(pos={self.data['position']}, type={self.data.get('piece')})"