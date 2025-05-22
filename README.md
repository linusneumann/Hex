# Systematischer Vergleich von Alpha-Beta-Suche mit einem Neuronalem Netzwerk anhand des Spiels Hex

This repsoitory contains the source code for the Bachelors thesis "Systematischer Vergleich von Alpha-Beta-Suche mit einem Neuronalem Netzwerk anhand des Spiels Hex".

The requirements for this project are: 
```
networkx
matplotlib (for better representation and for statistics)
numpy
tensorflow (only if you want to use the neural network)

```

To play a game of Hex against the computer use 
```
./play.sh [size]
```
Make sure you are in the correct directory!
This will launch a game with search depth 2 and size [size]. For playing against the CNN + ABS i recommend to play only on search depth 1.
Otherwise you will wait very long.

This was testet on Windows 11 with wsl.

The figures were created with the `graphs.py` file.
You will have to configure it yourself.

To run a tournament execute the file `tournament.py` file.

The dataset used for training is in 
 ```
actions/v4-11x11-mohex-mohex-cg2010-vs-mohex-mohex-weak.txt
```
