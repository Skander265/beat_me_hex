class Piece:
    def __init__(self, piece_type: str, owner=None):
        self.piece_type = piece_type
        self.owner = owner

    def get_type(self):
        return self.piece_type