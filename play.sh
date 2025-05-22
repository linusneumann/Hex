if [ -z "$1" ]; then
  echo "Usage: $0 <grid_size>"
  exit 1
fi

python3 -c "import hex_grid_networkx as hx; hex = hx.Hexgrid(2,$1,$1); hex.startgame()"