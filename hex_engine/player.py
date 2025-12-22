class Player:
    def __init__(self, piece_type: str, name: str = "Player"):
        self.piece_type = piece_type
        self.name = name

    def get_piece_type(self):
        return self.piece_type

    def get_name(self):
        return self.name