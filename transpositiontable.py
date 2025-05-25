from collections import namedtuple

TTEntry = namedtuple("TTEntry", ["value", "move", "depth", "flag"])

class TranspositionTable:
    def __init__(self):
        self.table = {}

    def store(self, zobrist_hash, value, move, depth, flag):
        self.table[zobrist_hash] = TTEntry(value, move, depth, flag)

    def retrieve(self, zobrist_hash, depth, alpha, beta):
        entry = self.table.get(zobrist_hash)
        if entry is None:
            return None

        if entry.depth < depth:
            return None  # zu flach → ignorieren

        # Auswertung nach Flag
        if entry.flag == "EXACT":
            return entry.value, entry.move
        elif entry.flag == "LOWERBOUND" and entry.value > beta:
            return entry.value, entry.move
        elif entry.flag == "UPPERBOUND" and entry.value < alpha:
            return entry.value, entry.move
        
        return None  # sonst nicht verwertbar
