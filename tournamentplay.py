import hex_grid_networkx as Hexgrid
import numpy as np
import os
import json
import time 

directoryrandom = "tournament/randomplay"
directoryabs = "tournament/absplay"
directoryrndabs = "tournament/rndabsplay"
directorycnn = "tournament/cnnplay"
directorycnnabs = "tournament/cnnabsplay"


def playgame(dir,player1,player2,size,index,randhistory,switch=False,tiefe1=None,tiefe2=None,board=Hexgrid.Hexgrid,time_limit=None):
    if randhistory == []:
        history = []
    else:
        history = randhistory
    
    times = []
    while(board.win == None):
        board.findwinner()
        if board.win != None:
            break
        start = time.time()
        if tiefe1 is not None:
            move =player1(1,suchtiefe=tiefe1,time_limit=time_limit)
        else:
            move =player1(1,time_limit=time_limit)
        end = time.time()
        times.append((1,end - start))
        board.findwinner()
        if board.win != None:
            break
        
        print(f"player 1 eval: {board.calcres()}")
        #print(f"player 1 eval: {board.calccnn(move)}")
        #board.displayboard()

        if move is not None:
            history.append((1,move))
        start = time.time()
        if tiefe2 is not None:
            move2 =player2(2,suchtiefe=tiefe2,time_limit=time_limit)
        else:
            move2 =player2(2,time_limit=time_limit)
        end = time.time()
        times.append((2,end - start))
        board.findwinner()
        if board.win != None:
            break

        print(f"player 2 eval: {board.calcres()}")
        #print(f"player 2 eval: {board.calccnn(move)}")
        #board.displayboard()

        if move2 is not None:
            history.append((2,move2))
        
        board.findwinner()
        if board.win != None:
            break
        # idk happens sometimes its a quick fix but not solution for the real problem
        if move and move2 == None: 
            board.displayboard()
            break

    history.append((board.win,'win'))
    writetofile(history,times,dir,size,index,switch) 

def randomtournament(rounds,size):
    workingdir = directoryrandom+str(size)
    open(workingdir,"w").close() #clear data before 
    print("Playing: " +str(rounds)+ " rounds")
    for i in range(rounds):
        board = Hexgrid.Hexgrid(3,size,size)
        playgame(workingdir,board.makerandommove,board.makecomputermove,size,i,rounds,board=board)
    print("switching sides")
    for i in range(rounds):
        board = Hexgrid.Hexgrid(3,size,size)
        playgame(workingdir,board.makecomputermove,board.makerandommove,size,i,rounds,board=board)

def writetofile(data,times,filename,size,game_index,switch=False):
    assert filename is not None
    current_page = 0
    if switch:
        current_page= current_page+1

    times = np.array(times)
    mean_1 = np.mean(times[times[:,0]==1][:,1])
    mean_2 = np.mean(times[times[:,0]==2][:,1])
    winner = data[-1][0]
    data = data[:-1] 
    output = {
        "win":winner,
        "size": size,
        "moves": [{"player": int(player), "move": list(move)}for player, move in data],
        "mean_times": {
            "player1": float(round(mean_1, 3)),
            "player2": float(round(mean_2, 3))
        }
    }

    if os.path.exists(filename+".json"):
        try:
            with open(filename+".json", "r") as f:
                all_data = json.load(f) 
        except json.JSONDecodeError:
            all_data = []
    else:
        all_data = []

    page_found = False
    for page in all_data:
        if page["page"] == current_page:
            page["games"].append(output)
            page_found = True
            break

    # Wenn keine Seite gefunden wurde, eine neue Seite erstellen und anhängen
    if not page_found:
        new_page = {
            "page": current_page,
            "games": [output]
        }
        all_data.append(new_page)

    with open(filename+".json","w") as f:
        json.dump(all_data,f,indent=2)


def makerandomboard(dir,board=Hexgrid.Hexgrid):
    steps = np.random.randint(5,10,dtype=int)
    player = 1
    moves = []

    for i in range(steps):
        move =(board.makerandommove(player))
        moves.append((player,move))
        player = board.nextplayer(player)
    return moves
        

def abstournament(rounds,size,depth1=None,depth2=None,time_limit=None):
    if depth1 is not None and depth2 is not None and time_limit is not None:
        workingdir = directoryrndabs+"-"+str(depth1)+"_"+str(depth2)+"_"+str(size)+"_"+str(time_limit)+"s"
    elif depth1 is not None and depth2 is not None:
        workingdir = directoryrndabs+"-"+str(depth1)+"_"+str(depth2)+"_"+str(size)+"(2)"
    else:
        workingdir = directoryrndabs+"-"+str(size)

    if depth1 is not None and depth2 is not None:
        open(workingdir+".json","w").close() #clear data before 
        print("Playing: " +str(rounds)+ " rounds")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size)
        
            randommoves = makerandomboard(workingdir,board)
            playgame(workingdir,board.makecomputermove,board.makecomputermove,size,i,randommoves,board=board,tiefe1=depth1,tiefe2=depth2,time_limit=time_limit)
        print("switching sides")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size)
            randommoves = makerandomboard(workingdir,board)
            playgame(workingdir,board.makecomputermove,board.makecomputermove,size,i,randommoves,switch=True,board=board,tiefe2=depth1,tiefe1=depth2,time_limit=time_limit)

    else:
        open(workingdir+".json","w").close() #clear data before 
        print("Playing: " +str(rounds)+ " rounds")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size)
            randommoves =makerandomboard(workingdir,board)
            playgame(workingdir,board.makecomputermove,board.makecomputermove,size,i,randommoves,board=board,time_limit=time_limit)
        print("switching sides")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size)
            randommoves=makerandomboard(workingdir,board)
            playgame(workingdir,board.makecomputermove,board.makecomputermove,size,i,randommoves,switch=True,board=board,time_limit=time_limit)

def cnntournament(rounds,size,depth1=None):
    if depth1 is not None :
        workingdir = directorycnnabs+"-"+str(depth1)+"_"+str(size)+"-norand"
        open(workingdir+".json","w").close() #clear data before  
        print("Playing: " +str(rounds)+ " rounds")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size,cnn=True)
            #randommoves=makerandomboard(workingdir,board)
            randommoves=[]
            playgame(workingdir,board.makecnnabsmove,board.makecomputermove,size,i,randommoves,board=board,tiefe2=depth1)
        print("switching sides")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size,cnn=True)
            #randommoves=makerandomboard(workingdir,board)
            randommoves=[]
            playgame(workingdir,board.makecomputermove,board.makecnnabsmove,size,i,randommoves,switch=True,board=board,tiefe2=depth1)
    else:
        workingdir = directorycnnabs+"-"+str(size)
        open(workingdir+".json","w").close() #clear data before 
        print("Playing: " +str(rounds)+ " rounds")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(1,size,size,cnn=True)
            #randommoves=makerandomboard(workingdir,board)
            randommoves=[]
            playgame(workingdir,board.makecnnabsmove,board.makecomputermove,size,i,randommoves,board=board)
            #playgame(workingdir,board.makecomputermove,board.makecnnmove,size,i,randommoves,switch=True,board=board)
        print("switching sides")
        for i in range(rounds):
            print("Playing game: "+str(i))
            board = Hexgrid.Hexgrid(2,size,size,cnn=True)
            #randommoves=makerandomboard(workingdir,board)
            randommoves = []
            playgame(workingdir,board.makecomputermove,board.makecnnabsmove,size,i,randommoves,switch=True,board=board)
            #playgame(workingdir,board.makecnnmove,board.makecomputermove,size,i,randommoves,switch=True,board=board)

if __name__ == "__main__":
    hg = Hexgrid.Hexgrid(2,6,6,cnn=False)
   
    #randomtournament(rounds=20,size=11)
    abstournament(10,size=7,depth1=2,depth2=3, time_limit=5)
    #randomtournament(5,8)
    #cnntournament(10,11,depth1=2)
   
    #ecken = rows,cols ; 0,cols ; 0,0 ;rows,0