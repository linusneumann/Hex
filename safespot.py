def makecomputermove(self,player,moveordering=True,suchtiefe=None,time_limit=None):
        list = None
        self.startplayer = player
        if self.win is None:
            start_time = time.time()
            moves = self.getMoves(player=player)
            if moveordering:
                h1,h2 = self.showboardeval() #get values from empyt board
                list = [x for _, x in sorted(zip(h1,moves ))] if player == 1 else [x for _, x in sorted(zip(h2, moves))] # perform move ordering
            bestzug = None
            if suchtiefe:
                tiefe = suchtiefe
            else:
                tiefe = self.Suchtiefe
            for i in range(1,tiefe+1):
                #time cheking
                if time_limit is not None and time.time() - start_time > time_limit:
                    break
                #calling abs
                wert,zug,history = self.minbeta(i,-np.inf, np.inf, player,zugliste=list,start_time=start_time,time_limit=time_limit) if player == 1 else self.maxbeta(i,-np.inf, np.inf, player,zugliste=list,start_time=start_time,time_limit=time_limit)
                #getting result and handling them
                np_history = np.array(history,dtype="object")

                if np_history.shape != ():
                    #len_moves = len(moves)
                    np_history = [x[0] for x in np_history[np.argsort([x[1] for x in np_history])]] # sort the array 
                    """if np_history.size != len_moves :
                        for move in moves:
                            if move not in [x[0] for x in np_history]:
                                np_history.append(move)"""
                    list = np_history
                else:
                    list = None

                if zug is not None:
                    bestzug = zug 
                    
            if bestzug is None:
                print("No move was found, something broke ?")  
            else:
                print(f"computer making move {bestzug}")
                self.makemove(bestzug,player) # have to unpack array/list
                self.ds_blue.history = []
                self.ds_red.history = [] #needed for consistency otherwise next undo would remove legal move
                return bestzug
        else:
            return None