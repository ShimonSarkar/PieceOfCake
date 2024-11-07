import os

default_requests = os.path.join("requests", "default", "easy.json")

possible_players = ["d"] + list(map(str, range(1, 12)))

# Commands for the simulator
INIT = 1
CUT = 2
ASSIGN = 3

timeout = 12000
