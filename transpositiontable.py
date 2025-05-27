from collections import namedtuple

TTEntry = namedtuple("TTEntry", ["value", "move", "depth", "flag"])

class TranspositionTable:
    def __init__(self):
        self.table = {}

    def store(self, zobrist_hash, value, move, depth, flag,player):
        self.table[(zobrist_hash,player)] = TTEntry(value, move, depth, flag)

    def retrieve(self, zobrist_hash, depth, alpha, beta,player):
        key = (zobrist_hash,player)
        entry = self.table.get(key)
        if entry is None:
            return None
        # not right depth
        if entry.depth < depth:
            return None  

        #get entries based on flag
        if entry.flag == "EXACT":
            return entry.value, entry.move, entry.flag
        elif entry.flag == "LOWERBOUND" and entry.value > alpha:
            alpha = entry.value
            #return entry.value, entry.move, entry.flag
        elif entry.flag == "UPPERBOUND" and entry.value < beta:
            beta = entry.value
            #return entry.valuen, entry.move, entry.flag
        if alpha >= beta:
            return entry.value, entry.move, entry.flag
        return None  
