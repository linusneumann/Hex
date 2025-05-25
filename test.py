import unittest
import numpy
from hex_grid_networkx import Hexgrid as hg

class TestHexgrid(unittest.TestCase):

    def setUp(self):
        self.hex  = hg(1,5,5)

    def test_make_move(self):
        
        self.hex.makemove((1,1),1)
        self.hex.makemove((2,2),2)
        self.assertEqual(self.hex.G.nodes[(1,1)]['resistance1'],self.hex.min_resistance,'Not the right value')
        self.assertEqual(self.hex.G.nodes[(1,1)]['resistance2'],self.hex.max_resistance,'Not the right value')
        self.assertEqual(self.hex.G.nodes[(2,2)]['resistance1'],self.hex.max_resistance,'Not the right value')
        self.assertEqual(self.hex.G.nodes[(2,2)]['resistance2'],self.hex.min_resistance,'Not the right value')

    def test_get_moves(self):
        moves = self.hex.getMoves(1)
        self.assertEqual(len(moves),(5*5),'Not all moves can be found')
        self.hex.makemove((1,1),1)
        moves = self.hex.getMoves(1)
        self.assertEqual(len(moves),(5*5)-1,'Not all moves can be found')

    def test_undomove(self):
        move = (1,1)
        self.hex.makemove(move,1)
        self.hex.undomove(move,1)
        self.assertEqual(self.hex.G.nodes[move]['resistance1'],1,'Not the right value')
        self.assertEqual(self.hex.G.nodes[move]['resistance2'],1,'Not the right value')

    def test_converttomatrix(self):
        move = (1,1)
        self.hex.makemove(move,1)
        np_array = self.hex.converttomatrix()
        self.assertEqual(np_array[move[0]][move[1]],1,'Not the right value')

    def test_findwinner(self):
        arr = [(0,0),(0,1),(0,2),(0,3),(0,4)]
        for move in arr:
            self.hex.makemove(move,1)
        self.hex.findwinner()
        self.assertEqual(self.hex.win,"red","No winner was found")

    def test_makemove_more(self):
        self.hex.makemove((2,0),1)
        self.hex.makemove((2,2),1)
    
        self.assertEqual(self.hex.ds_red.parent[(2,0)], self.hex.ds_red.parent[(5,5)])
        self.hex.makemove((2,1),1)
        self.assertEqual(self.hex.ds_red.parent[(2,0)], self.hex.ds_red.parent[(2,2)])

    def test_undomove_more(self):
        self.hex.makemove((2,0),1)
        self.hex.makemove((2,2),1)
    
        self.assertEqual(self.hex.ds_red.parent[(2,0)], self.hex.ds_red.parent[(5,5)])
        self.hex.makemove((2,1),1)
        self.hex.undomove((2,1),1)
        self.assertNotEqual(self.hex.ds_red.parent[(2,0)],self.hex.ds_red.parent[(2,2)])

if __name__ == '__main__':
    unittest.main()