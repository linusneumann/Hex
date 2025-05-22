# Systematischer Vergleich von Alpha-Beta-Suche mit einem Neuronalem Netzwerk anhand des Spiels Hex

This repsoitory contains the source code for the Bachelors thesis "Systematischer Vergleich von Alpha-Beta-Suche mit einem Neuronalem Netzwerk anhand des Spiels Hex".

The requirements for this project are: 
```
networkx
matplotlib
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
