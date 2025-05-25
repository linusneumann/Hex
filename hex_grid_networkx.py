
import itertools
import numpy as np, colorsys
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque # for finding winning path
from ast import literal_eval as make_tuple
import disjoint_set as ds
import copy
import sys
from collections import defaultdict
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Module Tensorflow is not installed\n You can't use the neural Network!")
import time
import transpositiontable as tt


class Hexgrid:
    def __init__(self,suchtiefe=2,rows=4,cols=4, cnn=False):
        self.G = nx.Graph()
        self.min_resistance = 1e-12 # benutzt für alles was für berechnungen 0 sein soll
        self.max_resistance = 1000000000
        self.rows = rows
        self.cols =cols
        self.Suchtiefe = suchtiefe
        self.ds_red =ds.DisjointSet([])
        self.ds_blue = ds.DisjointSet([])
        self.union_history =[]
        self.realnodes = {}
        self.top_node = (-1,-1)
        self.left_node = (-1,0)
        self.right_node = (rows,0)
        self.bott_node = (rows,rows)
        
        self.win = None
        self.specific_resistances = { }
        self.ztable = self.zobristtable()
        self.hashtable1 = tt.TranspositionTable()
        self.hashtable2 = tt.TranspositionTable()
        self.testarr= []
        self.besterzug = None
        self.startplayer=1
        self.history = defaultdict(int)
        self.G_hex = self.hexagonal_grid_graph(self.rows, self.cols, self.node_resistance, self.average_edge_resistance)
        if cnn:
            try:
                self.cnn = keras.models.load_model("4_64-30-cnntest13x13x5.keras")
            except ImportError:
                print("Could not load model. Tensorflow or Keras is not installed!")
        else:
            self.cnn = None
    

    def hexagonal_grid_graph(self,rows, cols, node_resistance_func, edge_resistance_func):
        
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]  # 6 Richtungen
        num = 1
        if rows != cols:
            print("nicht gültiges spielfeld")
            return
        # give every node their value
        node_resistances = {}
        for q, r in itertools.product(range(cols), range(rows)):
            node_resistances[(q, r)] = node_resistance_func((q, r))
        
        # add edges
        for q, r in itertools.product(range(cols), range(rows)):
            self.G.add_node((q, r), resistance1=node_resistances[(q, r)],name=num,resistance2=node_resistances[(q,r)])  # speichere knotengewicht
            

            num = num +1
            for dq, dr in directions:
                neighbor = (q + dq, r + dr)
                if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                    resistance1 = edge_resistance_func(node_resistances[(q, r)], node_resistances[neighbor])
                    resistance2 = edge_resistance_func(node_resistances[(q, r)],node_resistances[neighbor])
                    self.G.add_edge((q, r), neighbor, weight1=resistance1, weight2=resistance2)
                    

        self.ds_red = ds.DisjointSet(list(self.G.nodes))
        self.ds_blue = ds.DisjointSet(list(self.G.nodes))
        
        self.realnodes = copy.deepcopy(self.G.nodes)
        #player1
        self.G.add_node(self.top_node, resistance1=self.min_resistance,resistance2=self.max_resistance, name=0) # startknoten with resistance 0
        self.G.add_node(self.bott_node, resistance1=self.min_resistance,resistance2=self.max_resistance, name=17) # endknoten with resistance 0
        
        #player 2
        self.G.add_node(self.left_node, resistance1=self.max_resistance,resistance2=self.min_resistance, name=-1) # startknoten with resistance 0
        self.G.add_node(self.right_node, resistance1=self.max_resistance,resistance2=self.min_resistance, name=18) # endknoten with resistance 0
        

        self.ds_red.add_elements([self.bott_node,self.top_node])
        self.ds_blue.add_elements([self.right_node,self.left_node])
        for i in range(rows):
            self.G.add_edge(self.top_node, (i,rows-1), weight1=self.min_resistance,weight2=self.max_resistance)   #mit erster reihe verbunden
            self.G.add_edge(self.bott_node, (i,0), weight1=self.min_resistance,weight2=self.max_resistance) #mit letzter reihe verbunden
            self.G.add_edge(self.left_node, (0,i), weight1=self.max_resistance,weight2=self.min_resistance)   #mit erster reihe verbunden
            self.G.add_edge(self.right_node, (rows-1,i), weight1=self.max_resistance,weight2=self.min_resistance) #mit letzter reihe verbunden

            self.ds_red.union(self.bott_node, (i,0))
            self.ds_red.union(self.top_node,(i,rows-1))
            self.ds_blue.union(self.left_node,(0,i))
            self.ds_blue.union(self.right_node,(rows-1,i))
        
        self.ds_blue.history = []
        self.ds_red.history =[]        
        return self.G
    
    
    def node_resistance(self,node):
        return self.specific_resistances.get(node, 1)


    def average_edge_resistance(self,res1, res2):
        return max(round((res1 + res2), 2),self.min_resistance)

#
#----------------------------------------------------------------------------------------------------------------------------------------------
#
#   
#----------------------------------------------------------------------------------------------------------------------------------------------

    def setWeight(self,node,player,player1weight,player2weight,eval=False):
        if self.G.nodes[node] == None:
            return -1
        self.G.nodes[node]['resistance1'] = max(player1weight,self.min_resistance)
        self.G.nodes[node]['resistance2'] = max(player2weight,self.min_resistance)
        union_counter = 0
        for nbr,datadict in self.G.adj[node].items():
            nbr_res1 = self.G.nodes[nbr].get('resistance1', self.min_resistance)
            nbr_res2 = self.G.nodes[nbr].get('resistance2', self.min_resistance)

            resistance1 = self.average_edge_resistance(self.G.nodes[node]['resistance1'], self.G.nodes[nbr]['resistance1'])
            resistance2 = self.average_edge_resistance(self.G.nodes[node]['resistance2'], self.G.nodes[nbr]['resistance2'])
            self.G.edges[(node,nbr)]['weight1'] = max(resistance1,self.min_resistance)
            self.G.edges[(node,nbr)]['weight2'] = max(resistance2,self.min_resistance)
            """if self.G.nodes[nbr]['resistance1'] == player1weight  and self.G.nodes[nbr]['resistance2'] == player2weight and player != 3:
                if player ==1:
                    self.ds_red.union(nbr, node)
                elif player == 2:
                    self.ds_blue.union(nbr, node)
                else:
                    continue"""
            
                #print(f"Player: {player}, node: {node}, nbr: {nbr}, node_color: {self.G.nodes[node]['resistance1']}/{self.G.nodes[node]['resistance2']}")
               


            if player == 1 and self.G.nodes[node]['resistance1'] == player1weight and nbr_res1 == player1weight and player != 3 and eval is False:
                self.ds_red.union(nbr, node)
                union_counter +=1
            elif player == 2 and self.G.nodes[node]['resistance2'] == player2weight  and nbr_res2 == player2weight and player != 3 and eval is False:
                self.ds_blue.union(nbr, node)
                union_counter +=1
        return union_counter
    

    def getOhm(self,name):
        for node in self.G.nodes:
            if(self.G.nodes[node]['name'] == name):
                return self.G.nodes[node]['resistance1'],self.G.nodes[node]['resistance2']
        return -1

    def getMoves(self,player):
        arr= []
        if player == 1:
            for node in self.G.nodes:
                if(self.G.nodes[node]['resistance1'] == 1):
                    #arr.append(G.nodes[node]['name'])
                    arr.append(node)
            
        if player == 2:
            for node in self.G.nodes:
                if(self.G.nodes[node]['resistance2'] == 1):
                    #arr.append(G.nodes[node]['name'])
                    arr.append(node)

        return sorted(arr, key=lambda m:-self.history[m])
    
    def getName(self,node):
        return self.G.nodes[node]['name']
    
    def nextplayer(self,player):
        nextplayer ={
            1:2,
            2:1
        }
        return nextplayer.get(player)

    def makemove(self,move,player,eval=False):
        if player ==1: # red / minimizing i think it makes sense right now 
            error=self.setWeight(move,player,self.min_resistance,self.max_resistance,eval)
            self.union_history.append(error)
            if(error == -1):
                print("setWeight ist kaputt")
        elif player ==2: #blue / maximazing
            error=self.setWeight(move,player,self.max_resistance,self.min_resistance,eval)
            self.union_history.append(error)
            if(error == -1):
                print("setWeight ist kaputt")
        else:
            print("player ist falsch: "+ player )
        

    def undomove (self,move,player):
        self.setWeight(move,3,1,1)
        num_union = self.union_history.pop()
        for i in range(num_union):
        #self.win = None
            if player ==1:
                self.ds_red.undo_union()
            elif player == 2:
                self.ds_blue.undo_union()
            else:
                print("player ist falsch: "+ player )
        self.win = None
        self.findwinner()


    def printnodes(self):
        for node, data in self.G_hex.nodes(data=True):
            print(f"Knoten {node}: {data['resistance1']} Ohm,{data['resistance2']} Ohm, Name = {data['name']}")
        return

    def converttomatrix(self):
        arr=np.zeros((self.rows,self.cols),dtype=int)
        for node in self.realnodes:
            if self.G.nodes[node]['resistance1'] == self.min_resistance:
                arr[node[0]][node[1]] = 1
            elif self.G.nodes[node]['resistance1'] == self.max_resistance:
                arr[node[0]][node[1]] = -1
            else:
                continue
        return arr

    def calcres(self):
        # red / blue
        self.red_dist = nx.resistance_distance(self.G_hex, self.bott_node, self.top_node, weight='weight1') 
        self.blue_dist = nx.resistance_distance(self.G_hex, self.left_node, self.right_node, weight='weight2') 
        """if self.startplayer != 1:
            print("startplayer = "+str(self.startplayer))"""

        if self.startplayer == 1:
            return np.log(self.red_dist /self.blue_dist)
        if self.startplayer == 2:
            #return self.blue_dist/self.red_dist
            return np.log(self.blue_dist /self.red_dist) # a little bit sus 
        
    def calccnn(self,move):
        player1 = self.makecnneval(1,move)
        #player2 = self.makecnneval(2,move)
        logit_pred = np.log(max(player1,self.min_resistance)/(1-max(player1,self.min_resistance)))
        return logit_pred

    def findwinner(self):#min_resistance*rows*cols
        if self.ds_red.find(self.top_node) == self.ds_red.find(self.bott_node):
            self.win = "red"
            #print("found red win ")
            return
        if self.ds_blue.find(self.left_node) == self.ds_blue.find(self.right_node):
            self.win = "blue"
            #print("found blue win ")
            return

    def findpath(self,source, target,max_cost =1):
        queue = deque([(source, [source], 0)])  # (aktueller Knoten, Pfad, aktuelle Kosten) idk about that one 
        visited = set()  # Vermeidung von Zyklen
        visited.add(source)
        
        while queue:
            node, path, cost = queue.popleft()
            
            if node == target:
                return path  # Ersten gefundenen gültigen Pfad zurückgeben
            
            for neighbor in self.G.neighbors(node):
                edge_weight = self.G[node][neighbor].get("weight1", float(1))  # Standardgewicht ∞, falls nicht vorhanden
                
                if edge_weight <= max_cost and neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + edge_weight))
                    visited.add(neighbor)
    
        return []  # Kein gültiger Pfad gefunde
    
    #Only for show purpose 
    def hex_pos(self,G):
        
        pos = {}
        for (q, r) in self.G.nodes():
            x = q + 0.5 * r
            y = r * np.sqrt(3) / 2
            pos[(q, r)] = (x, y)

        pos[self.top_node] = (self.rows-1,self.rows-1)
        pos[self.bott_node] = (round(self.rows/2),-1)
        pos[self.left_node]= (-1,round(self.rows/2))
        pos[self.right_node]= (self.rows+1,1.5)
        return pos
    
    #makes eval for every position on board for both players for abs players
    def showboardeval(self):
        arr1 = []
        arr2 = []
        start_temp =self.startplayer
        for node in self.getMoves(player=1):
            self.makemove(node,player=1,eval=True)
            self.startplayer = 1
            val1 = round(self.calcres(),3)
            arr1.append(val1)
            self.undomove(node,player=1)
            #should be the same for player2
            self.makemove(node,player=2,eval=True)
            self.startplayer = 2
            val1 = round(self.calcres(),3)
            arr2.append(val1)
            self.undomove(node,player=2)

        self.startplayer = start_temp
        return arr1, arr2
    #makes eval for every position on board for both players for cnn players
    def showboardeval2(self,tiefe=1):
        arr1 = []

        for node in self.getMoves(player=1):
            self.makemove(node,player=1)
            val1 = round(self.calccnn(node),5)
            arr1.append(val1)
            self.undomove(node,player=1)
            #should be the same for player2

        return arr1
    
    def showboardeval3(self):
        arr1 = []

        for node in self.getMoves(player=1):
            self.makemove(node,player=1)
            val1 = round(self.calcres(player=1),3)
            arr1.append((val1,node))
            self.undomove(node,player=1)
           

        return arr1

    def zobristtable(self):
        table = np.random.randint(2**31-1,size=(2, self.cols*self.rows),dtype=int) # ideally have arr of random numbers with no ovelap
        #lets say index 0 = player 1 and index 1 ist player 2
        return table
    
    def hashboard(self):
        hash = 0
        for i,node in enumerate(self.realnodes):
            if self.G.nodes[node]['resistance1'] == self.min_resistance:
                hash = hash ^ self.ztable[0][i]
            if self.G.nodes[node]['resistance2'] == self.max_resistance:
                hash = hash ^ self.ztable[1][i]
        
        return hash

    def move_heuristic(self, zug):
        #isnt really used anymore
        x, y = zug
        # prioritize close to middle
        mx, my = self.rows // 2, self.cols // 2
        return -((x - mx)**2 + (y - my)**2)

    def maxbeta(self,tiefe,alpha, beta, player, zugliste=None, start_time=None, time_limit=None):
        if start_time is None:
            start_time = time.time()
        self.findwinner()
        if tiefe == 0  or self.win != None: 
            """if self.win == "red":
                return float(self.max_resistance),None,None if self.startplayer == 1 else -float(self.max_resistance)
            elif self.win == "blue":
                return float(self.max_resistance),None,None if self.startplayer == 2 else -float(self.max_resistance)"""
            
            return self.calcres(), None,None
        
        # transposition table makes it faster 
        hash = self.hashboard()
        history = []

        if self.startplayer == 1:
            tt_result = self.hashtable1.retrieve(hash, tiefe, alpha, beta)
        else:
            tt_result = self.hashtable2.retrieve(hash, tiefe, alpha, beta)
        if tt_result is not None:
            history.append((tt_result[1],tt_result[0]))
            return tt_result[0],tt_result[1], history
        
        if zugliste is None: # we dont get zugliste 
            zugliste = self.getMoves(player)

        besterzug = None
        maxWert = -np.inf
        

        for zug in zugliste:
            if time_limit is not None and time.time() - start_time > time_limit:
                return maxWert, besterzug, history  # time limit reached
            
            self.makemove(zug,player)
            wert,_ ,_ = self.minbeta(tiefe-1,maxWert, beta,self.nextplayer(player),start_time=start_time,time_limit=time_limit)  # make player differ 
            self.undomove(zug,player)
            
            history.append((zug,wert))

            if (wert > maxWert):
                maxWert = wert
                besterzug = zug 
                if (maxWert >= beta):
                    break

            # flag bestimmen
        if maxWert <= alpha:
            flag = "LOWERBOUND"
        elif maxWert >= beta:
            flag = "UPPERBOUND"
        else:
            flag = "EXACT"

        if player == 1:
            self.hashtable1.store(hash,maxWert,besterzug,tiefe,flag)
        else:
            self.hashtable2.store(hash,maxWert,besterzug,tiefe,flag)
        
        return maxWert,besterzug,history
 
    def minbeta(self, tiefe,alpha, beta, player, zugliste=None, start_time=None, time_limit=None):
        if start_time is None:
            start_time = time.time()
        self.findwinner()
        
        if tiefe == 0  or self.win != None:
            return self.calcres(), None,None
            """ if self.win == "red":
                return -float(self.max_resistance),None,None if self.startplayer == 1 else float(self.max_resistance)
            elif self.win == "blue":
                return -float(self.max_resistance),None,None if self.startplayer == 2 else -float(self.max_resistance)"""
            
        
        
        hash = self.hashboard()
        history = []

        if self.startplayer == 1:
            tt_result = self.hashtable1.retrieve(hash, tiefe, alpha, beta)
        else:
            tt_result = self.hashtable2.retrieve(hash, tiefe, alpha, beta)

        if tt_result is not None:
            history.append((tt_result[1],tt_result[0]))
            return tt_result[0],tt_result[1], history
        
        
        if zugliste is None: # we dont get zugliste 
            zugliste = self.getMoves(player)


        besterzug = None
        minWert = np.inf
        

        for zug in zugliste:
            if time_limit is not None and time.time() - start_time > time_limit:
                return minWert, besterzug, history  # time limit reached

            self.makemove(zug,player)
            wert, _ ,_= self.maxbeta(tiefe-1,alpha, minWert,self.nextplayer(player),start_time=start_time,time_limit=time_limit)
            self.undomove(zug,player)

            history.append((zug,wert))

            if wert < minWert:
                minWert = wert
                besterzug = zug
                if (minWert <= alpha):
                    break
                       
            # flag bestimmen
        if minWert <= alpha:
            flag = "UPPERBOUND"
        elif minWert >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"
        
       
        if player == 1:
            self.hashtable1.store(hash,minWert,besterzug,tiefe,flag)
        else:
            self.hashtable2.store(hash,minWert,besterzug,tiefe,flag)
        
        return minWert,besterzug,history
    
    def maxbeta_cnn(self,tiefe,alpha, beta,player,move=None,zugliste=None):
            self.findwinner()
            if tiefe == 0  or self.win != None: 
                if self.win == "red":
                    return float(self.max_resistance),None,None if self.startplayer == 1 else -float(self.max_resistance)
                elif self.win == "blue":
                    return float(self.max_resistance),None,None if self.startplayer == 2 else -float(self.max_resistance)
                return self.calccnn(move), None,None
            
            hash = self.hashboard()
            
            if self.startplayer == 1:
                tt_result = self.hashtable1.retrieve(hash, tiefe, alpha, beta)
            else:
                tt_result = self.hashtable2.retrieve(hash, tiefe, alpha, beta)
            if tt_result is not None:
                return tt_result[0],tt_result[1], None
            
            if zugliste is None: # we dont get zugliste 
                zugliste = self.getMoves(player)

            besterzug = None
            maxWert = -np.inf
            history = []

            for zug in zugliste: 
                self.makemove(zug,player)
                wert,_,_ = self.minbeta_cnn(tiefe-1,maxWert, beta,self.nextplayer(player),zug)  # make player differ 
                self.undomove(zug,player)

                history.append((zug,wert))

                if (wert > maxWert):
                    maxWert = wert
                    besterzug = zug
                    if (maxWert >= beta): 
                        break

                # flag bestimmen
            if maxWert <= alpha:
                flag = "LOWERBOUND"
            elif maxWert >= beta:
                flag = "UPPERBOUND"
            else:
                flag = "EXACT"
            
            if player == 1:
                self.hashtable1.store(hash,maxWert,besterzug,tiefe,flag)
            else:
                self.hashtable2.store(hash,maxWert,besterzug,tiefe,flag)
            
            return maxWert,besterzug,history

    def minbeta_cnn(self, tiefe,alpha, beta,player,move=None,zugliste=None):
        self.findwinner()
        if tiefe == 0  or self.win != None: 
                if self.win == "red":
                    return float(self.max_resistance),None,None if self.startplayer == 1 else -float(self.max_resistance)
                elif self.win == "blue":
                    return float(self.max_resistance),None,None if self.startplayer == 2 else -float(self.max_resistance)
                return self.calccnn(move), None,None
        
        
        hash = self.hashboard()
        
        if self.startplayer == 1:
            tt_result = self.hashtable1.retrieve(hash, tiefe, alpha, beta)
        else:
            tt_result = self.hashtable2.retrieve(hash, tiefe, alpha, beta)

        if tt_result is not None:
            return tt_result[0],tt_result[1], None
        
        
        if zugliste is None: # we dont get zugliste 
            zugliste = self.getMoves(player)


        besterzug = None
        minWert = np.inf
        history = []

        for zug in zugliste:
            self.makemove(zug,player)
            wert, _ ,_= self.maxbeta_cnn(tiefe-1,alpha, minWert,self.nextplayer(player),zug)
            self.undomove(zug,player)

            history.append((zug,wert))

            if wert < minWert:
                minWert = wert
                besterzug = zug
                if (minWert <= alpha):
                    break
                       
            # flag bestimmen
        if minWert <= alpha:
            flag = "UPPERBOUND"
        elif minWert >= beta:
            flag = "LOWERBOUND"
        else:
            flag = "EXACT"
        
        if player == 1:
            self.hashtable1.store(hash,minWert,besterzug,tiefe,flag)
        else:
            self.hashtable2.store(hash,minWert,besterzug,tiefe,flag)
        
        return minWert,besterzug,history
    #returns colors for visualization
    def choose_colors(self,node_list):
        num_colors = len(node_list)
        colors=[]
        for i in range(num_colors):
            if(self.G.nodes[node_list[i]]['resistance1'] == self.max_resistance):
                colors.append((47/255, 152/255, 245/255)) #black piece view from first person who starts ig
            elif(self.G.nodes[node_list[i]]['resistance1'] == 1):
                colors.append((219/255, 167/255, 99/255))  #no piece 
            else:
                colors.append((227/255, 74/255, 18/255))  #white piece 
        return colors
         
    #shows the board as graph, arr1 is heuristik for player1 arr2 is heuristik for player2 both are optional
    def displayboard(self,arr1 = [],arr2=[]):
        if len(arr1) == 0 and len(arr2) == 0:
            pos = self.hex_pos(self.G_hex)
            colors = self.choose_colors(list(self.G))
            nx.draw(self.G_hex, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=500, font_size=10)
            plt.title("Hex grid")
            plt.show()
        elif len(arr1) !=0 and len(arr2) == 0:
            pos = self.hex_pos(self.G_hex)
            colors = self.choose_colors(list(self.G))
            labels = {}
            for i,node in enumerate(self.getMoves(player=1)):
                labels[node] = arr1[i]
            nx.draw(self.G_hex, pos, with_labels=True,labels=labels, node_color=colors, edge_color="gray", node_size=500, font_size=10)
            plt.title("player1 eval")
            plt.show()
        elif len(arr2) !=0 and len(arr1) == 0:
            pos = self.hex_pos(self.G_hex)
            colors = self.choose_colors(list(self.G))
            labels = {}
            for i,node in enumerate(self.getMoves(player=2)):
                labels[node] = arr2[i]
            nx.draw(self.G_hex, pos, with_labels=True,labels=labels, node_color=colors, edge_color="gray", node_size=500, font_size=10)
            plt.title("player2 eval")
            plt.show()
        elif len(arr2) !=0 and len(arr1) !=0:
            pos = self.hex_pos(self.G_hex)
            colors = self.choose_colors(list(self.G))
            fig, axes = plt.subplots(2)
            labels = {}
            labels2 = {}
            for i,node in enumerate(self.getMoves(player=1)):
                labels[node] = arr1[i]
                labels2[node] = arr2[i]
            nx.draw(self.G_hex, pos, with_labels=True,labels=labels, node_color=colors, edge_color="gray", node_size=500, font_size=10, ax=axes[0])
            nx.draw(self.G_hex, pos, with_labels=True,labels=labels2, node_color=colors, edge_color="gray", node_size=500, font_size=10, ax=axes[1])

            plt.title("player1 and player2 eval")
            fig.tight_layout()
            plt.show()
        else:
            return
 
    #suchtiefe and time_limit are not used in here, only for tournament purpose
    def makeplayermove(self,player,suchtiefe=None,time_limit=None):
        if 'matplotlib' not in sys.modules:
            arr = self.converttomatrix()
            print("  "+ np.array_str(np.arange(arr.shape[0])))
            for i in range(arr.shape[0]):
                print(i,arr[i])
        else:
            self.displayboard()
            
        finished = False
        while(not finished):
            move= input("Make move: format\"(x,x)\" ").strip()

            if "stop" in move:
                print("Exiting programm...")
                sys.exit(0)
            try:
                tuple = make_tuple(move)
            except ValueError:
                print("Input was not a tuple")
                continue
            if not isinstance(tuple[0],int):
                print("first Value was not a number")
                continue
            if not isinstance(tuple[1],int):
                 print("second Value was not a number")
                 continue
            if tuple[0] >= self.rows or tuple[0] <0 or tuple[1] >= self.rows or tuple[1] <0:
                print("Input was not in correct coordinates")
                continue
            finished = True

        self.makemove(tuple,player) # make move
        self.ds_blue.history = []
        self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move 
        return move
    
    def makerandommove(self,player):
        moves = self.getMoves(player)
        if len(moves) <=0:
            return None
        randnr = np.random.randint(len(moves),dtype=int)
        self.makemove(moves[randnr],player)
        self.ds_blue.history = []
        self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move 
        return moves[randnr]
    

    
    def makecnnmove(self,player,suchtiefe=None,time_limit=None):
        if self.win is None:
            validmoves = self.getMoves(player)
            prediction = None
            if self.cnn:
                board = self.converttomatrix()
                board.astype(dtype=np.float32)
                #for 13x13x5
                padded_board = add_padding(board)
                only_black = add_padding((board ==1) * 1)
                only_white = add_padding((board == -1) * -1)
                #for 11x11x3 switch board and padded_board
                color_plane = np.ones_like(padded_board) * (1 if player == 1 else -1)
                next_player_plane = np.ones_like(padded_board) * (1 if player == 1 else -1)
                input_planes = np.stack([padded_board,only_black,only_white, color_plane, next_player_plane], axis=-1)

                tensor = tf.convert_to_tensor(input_planes[None, ...], dtype=tf.float32)
                raw = self.cnn.predict(tensor)
                #print(raw)
                prediction = np.argsort(*raw)[::-1][:10]    #look at the 10 highes rated moves

            for pred in prediction:
                row, col = divmod(pred,11)
                res = [move for move in validmoves if (row.item(),col.item()) == move]
                if res :
                    self.makemove((row.item(),col.item()),player)
                    self.ds_blue.history = []
                    self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move 
                    return (row.item(),col.item())
                
            print(f"Cnn had no answer. Resorting to random move...")    #for case when NN wants to make a illegal move
            return self.makerandommove(player)
        else:
            return None
    
    def makecomputermove(self,player,moveordering=True,suchtiefe=None,time_limit=None):
        list = None
        self.startplayer = player
        if self.win is None:
            start_time = time.time()
            moves = self.getMoves(player=player)
            h1,h2 = None, None
            if moveordering:
                h1,h2 = self.showboardeval() #get values from empyt board
                list = [x for _, x in sorted(zip(h1,moves ))] if player == 1 else [x for _, x in sorted(zip(h2, moves))] # perform move ordering

            index = None
            if list is not None and len(list) >0:
                bestzug = list[0] #if we have eval then use it
                index = moves.index(bestzug) 
            else:
                bestzug =  None

            bestval = 0
            if h1 is not None and h2 is not None and index is not None:
                bestval = h1[index] if player == 1 else h2[index]

            if suchtiefe:
                tiefe = suchtiefe
            else:
                tiefe = self.Suchtiefe
            
            """if bestval < -22:
                print(f"computer making move {bestzug}")
                self.makemove(bestzug,player) # have to unpack array/list
                self.ds_blue.history = []
                self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move
                return bestzug"""
           
            for i in range(1,tiefe+1):
                #time cheking
                if time_limit is not None and time.time() - start_time > time_limit:
                    break
                #calling abs
                wert,zug,history = self.minbeta(i,-np.inf, np.inf, player,zugliste=list,start_time=start_time,time_limit=time_limit) 
                #getting result and handling them
                np_history = np.array(history,dtype="object")

                if np_history.shape != ():
                    #len_moves = len(moves)
                    #sort the moves based on iterative deepening and heuristic
                    eval_dict = {tuple(move): eval for move, eval in np_history}
                    sorted_dict = sorted(eval_dict.items(),key=lambda x: x[1])
                    sorted_moves = [move for move,_ in sorted_dict]
                    remaining = [move for move in moves if move not in eval_dict]
                    final_ordering = sorted_moves + remaining 
                    list = final_ordering
                else:
                    list = None

                if zug is not None:
                    #if bestval > -22:
                    bestzug = zug 
                    
            if bestzug is None:
                print("No move was found, something broke ?") 
                return None
            else:
                print(f"computer making move {bestzug}")
                self.makemove(bestzug,player) 
                self.ds_blue.history = []
                self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move
                return bestzug
        else:
            return None
        
    def makecnnabsmove(self,player,moveordering=True,suchtiefe=None,time_limit=None):
        list = None
        self.startplayer = player
        if self.win is None:
            moves = self.getMoves(player=player)
            if moveordering:
                h1 = self.showboardeval2() #get values from empyt board
                list = [x for _, x in sorted(zip(h1, moves))] # perform move ordering
                list = list[::-1] # we want high rated moves in the front
            bestzug = None

            if suchtiefe:
                tiefe = suchtiefe # for tournament purposes
            else:
                tiefe = self.Suchtiefe
            for i in range(1,tiefe+1):
                wert,zug,history = self.maxbeta_cnn(i,np.inf, -np.inf, player,zugliste=list) 
                np_history = np.array(history,dtype="object")

                if np_history.shape != ():
                    #len_moves = len(moves)
                    #sort the moves based on iterative deepening and heuristic
                    eval_dict = {tuple(move): eval for move, eval in np_history}
                    sorted_dict = sorted(eval_dict.items(),key=lambda x: x[1])
                    sorted_moves = [move for move,_ in sorted_dict]
                    remaining = [move for move in moves if move not in eval_dict]
                    final_ordering = sorted_moves + remaining 
                    list = final_ordering
                else:
                    list = None

                if zug is not None:
                    bestzug = zug 
                    
            if bestzug is None:
                print("No move was found, something broke ?") 
                return None
            else:
                print(f"computer making move {bestzug}")
                self.makemove(bestzug,player) 
                self.ds_blue.history = []
                self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move
                return bestzug
        else:
            return None    

    def makecnneval(self,player,move,suchtiefe=None):
        if self.win is None:
            #validmoves = self.getMoves(player)
            prediction = None
            if self.cnn:
                board = self.converttomatrix()
                board.astype(dtype=np.float32)
                #for 13x13x5
                padded_board = add_padding(board)
                only_black = add_padding((board ==1) * 1)
                only_white = add_padding((board == -1) * -1)
                #for 11x11x3 switch board and padded_board
                color_plane = np.ones_like(padded_board) * (1 if player == 1 else -1)
                next_player_plane = np.ones_like(padded_board) * (1 if player == 1 else -1)
                input_planes = np.stack([padded_board,only_black,only_white, color_plane, next_player_plane], axis=-1)

                tensor = tf.convert_to_tensor(input_planes[None, ...], dtype=tf.float32)
                raw = self.cnn.predict(tensor,verbose=0)[0]

                move11 = move[0] * 11 + move[1]
                prediction = raw[move11]
                
                
            return prediction
        else:
            return None

    def startgame(self):
        assert(self.G_hex != None)
        player = 1
        computer = 2
        finished = False
        first = True
        self.ztable = self.zobristtable()
        while not finished:
            print("Which player would you like to play : \"1\",\"2\" (1 starts)")
            playerchoice =input()
            match playerchoice:
                case "1":
                    player = 1 
                    computer = 2
                    finished = True
                case "2":
                    player = 2
                    computer = 1
                    finished = True
                case _:
                    print("no such argument allowed")
        
        
        while(self.win == None):
            if self.win != None:
                    print(f"{self.win} hat gewonnen")
                    break
            print('player1 start:') # player1 /red
            if player ==1:
                self.makeplayermove(player)
                if self.win != None:
                    break
                print("computer move")  #player2 /blue
                if self.cnn:
                    self.makecnnabsmove(player=computer)
                else:
                    self.makecomputermove(player=computer)
                    print(self.startplayer)
                    print(self.calcres())
                if self.win != None:
                    break
            if player ==2:
                print("computer move")  #player2 /blue
                if first == True:
                    if self.cnn:
                        self.makecnnabsmove(player=computer)
                    else:
                        self.makecomputermove(player=computer)
                        print(self.startplayer)
                        print(self.calcres())
                    #self.makemove((int(self.rows/2),int(self.cols/2)),player=1)
                    #self.makecnnmove(computer)
                    first = False
                else:
                    if self.cnn:
                        self.makecnnabsmove(player=computer)
                    else:
                        self.makecomputermove(player=computer)
                        print(self.startplayer)
                        print(self.calcres())
                    if self.win != None:
                        break
                self.makeplayermove(player)
                if self.win != None:
                    break
            self.findwinner() 
            #player 1/red
            #player 2 /blue   
        print(f"{self.win} won !")
    #test function        
    def test(self,player):
        foo = " "
        while(self.win == None ):
            #move = make_tuple(self.makeplayermove(player=1))
            move = self.makecomputermove(1,suchtiefe=2)

            #self.startplayer=2
            print(self.startplayer)
            #print(self.calccnn(move=move))
            self.displayboard()
            labels1,labels2=self.showboardeval()
            self.findwinner() 
            if self.win != None:
                print(f"{self.win} hat gewonnen")
                break

            foo = input("next?")
            foo.strip()
            foo.strip('\n')
            self.findwinner() 
            if "show" in foo:
                self.displayboard(arr1=labels1,arr2=labels2)
                pass
            
            #self.makeplayermove(player=1)
            self.makecomputermove(2,suchtiefe=3)
            self.displayboard()
            #print(self.startplayer)
            #print(self.calcres())
            self.findwinner() 
            #player 1/red
            #player 2 /blue   
            
            
            if foo =="s":
                break

      
def add_padding(board):
        padded_board = np.zeros((13,13),dtype=int)
        padded_board[1:-1,1:-1] = board

        #padding
        #oben und unten
        padded_board[0,1:-1] = 1
        padded_board[-1,1:-1] = 1
        #links und rechts
        padded_board[1:-1,0] = -1
        padded_board[1:-1,-1] = -1
        #ecken
        padded_board[0,0] = 1 
        padded_board[12,12] = 1 
        padded_board[0,12] = 1 
        padded_board[12,0] = 1 
        #print(padded_board)
        return padded_board   
def makerandomboard(dir,board=Hexgrid):
    steps = np.random.randint(5,10,dtype=int)
    player = 1
    moves = []

    for i in range(steps):
        move =(board.makerandommove(player))
        moves.append((player,move))
        player = board.nextplayer(player)
    return moves

if __name__ == "__main__":
    hg = Hexgrid(2,5,5,cnn=False)
    #hg.startgame()
    #randommoves = makerandomboard("tournament/hallo123",hg)
    #hg.makeplayermove(1)
    #hg.test(1)
    #hg.makeplayermove(1)
    #arr1 = hg.showboardeval2()
    hg.displayboard()
    #ecken = rows,cols ; 0,cols ; 0,0 ;rows,0
