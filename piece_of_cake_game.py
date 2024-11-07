import json
import os
import sys
import time
import numpy as np
import math
import traceback
import matplotlib.pyplot as plt

from shapely import points, centroid

import miniball

from piece_of_cake_state import PieceOfCakeState
from constants import *
import constants
from utils import *
from players.default_player import Player as DefaultPlayer
from players.G2_Player import G2_Player
from players.g6_player import Player as G6_Player
from players.g1_player import Player as G1_Player
from players.g8_player import G8_Player
from players.group10_player import Player as G10_Player
from players.player_7 import Player as G7_Player
from players.g9_player import Player as G9_Player
from players.g5_player import Player as G5_Player
from players.group_3 import Player as G3_Player
from players.g4_player import Player as G4_Player
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import tkinter as tk


class PieceOfCakeGame:
    def __init__(self, args, root):
        self.start_time = time.time()
        self.use_gui = not args.no_gui
        self.do_logging = not args.disable_logging
        self.is_paused = False
        self.root = root
        self.game_state = "pause"
        self.game_speed = "normal"
        self.scale = int(args.scale)

        if self.use_gui:
            self.canvas_width = 1500
            self.canvas_height = 800
            self.use_timeout = False
        else:
            self.use_timeout = not args.disable_timeout

        self.logger = logging.getLogger(__name__)
        # create file handler which logs even debug messages
        if self.do_logging:
            self.logger.setLevel(logging.DEBUG)
            self.log_dir = args.log_path
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dir, 'debug.log'), mode="w")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('%(message)s'))
            fh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(fh)
            result_path = os.path.join(self.log_dir, "results.log")
            rfh = logging.FileHandler(result_path, mode="w")
            rfh.setLevel(logging.INFO)
            rfh.setFormatter(logging.Formatter('%(message)s'))
            rfh.addFilter(MainLoggingFilter(__name__))
            self.logger.addHandler(rfh)
        else:
            if args.log_path:
                self.logger.setLevel(logging.INFO)
                result_path = args.log_path
                self.log_dir = os.path.dirname(result_path)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                rfh = logging.FileHandler(result_path, mode="w")
                rfh.setLevel(logging.INFO)
                rfh.setFormatter(logging.Formatter('%(message)s'))
                rfh.addFilter(MainLoggingFilter(__name__))
                self.logger.addHandler(rfh)
            else:
                self.logger.setLevel(logging.ERROR)
                self.logger.disabled = True

        self.logger.info("Initialise random number generator with seed {}".format(args.seed))

        self.rng = np.random.default_rng(args.seed)

        self.player = None
        self.player_name = None
        self.player_time = constants.timeout
        self.player_timeout = False
        self.cake_len = None
        self.cake_width = None

        self.tolerance = args.tolerance
        self.requests = []
        self.cur_pos = None
        self.prev_pos = None
        self.penalty = None
        self.polygon_list = None
        self.goal_reached = False
        self.assignment = None
        self.x_offset = 50
        self.y_offset = 50
        self.cake_cuts = []
        self.turns = 0
        self.valid_moves = 0
        self.timeout_warning_count = 0
        self.max_turns = 1e9

        self.add_player(args.player)
        self.initialize(args.requests)

    def add_player(self, player_in):
        if player_in in constants.possible_players:
            if player_in.lower() == 'd':
                player_class = DefaultPlayer
                player_name = "Default Player"
            else:
                player_class = eval("G{}_Player".format(player_in))
                player_name = "Group {}".format(player_in)

            self.logger.info(
                "Adding player {} from class {}".format(player_name, player_class.__module__))
            precomp_dir = os.path.join("precomp", player_name)
            os.makedirs(precomp_dir, exist_ok=True)

            start_time = 0
            is_timeout = False
            try:
                start_time = time.time()
                player = player_class(rng=self.rng, logger=self.get_player_logger(player_name),
                                      precomp_dir=precomp_dir, tolerance=self.tolerance)
            except TimeoutException:
                is_timeout = True
                player = None
                self.logger.error(
                    "Initialization Timeout {} since {:.3f}s reached.".format(player_name, constants.timeout))

            init_time = time.time() - start_time

            if not is_timeout:
                self.logger.info("Initializing player {} took {:.3f}s".format(player_name, init_time))
            self.player = player
            self.player_name = player_name

        else:
            self.logger.error("Failed to insert player {} since invalid player name provided.".format(player_in))

    def get_player_logger(self, player_name):
        player_logger = logging.getLogger("{}.{}".format(__name__, player_name))

        if self.do_logging:
            player_logger.setLevel(logging.INFO)
            # add handler to self.logger with filtering
            player_fh = logging.FileHandler(os.path.join(self.log_dir, '{}.log'.format(player_name)), mode="w")
            player_fh.setLevel(logging.DEBUG)
            player_fh.setFormatter(logging.Formatter('%(message)s'))
            player_fh.addFilter(PlayerLoggingFilter(player_name))
            self.logger.addHandler(player_fh)
        else:
            player_logger.setLevel(logging.ERROR)
            player_logger.disabled = True

        return player_logger

    def initialize(self, requests):
        # If requests are provided, load them into self.requests
        if requests:
            self.logger.info("Loading requests from {}".format(requests))
            with open(requests, "r") as f:
                requests_obj = json.load(f)
            self.requests = np.array(requests_obj["requests"]).tolist()

            # Validate the map
            if not self.validate_requests():
                self.logger.error("Requests are invalid")
                raise Exception("Invalid Requests")
        else:
            # If no requests are provided, generate requests uniformly from the range 10 to 100
            self.generate_requests()

        print("Requests generated successfully...")

        # Uncomment to save the maze in a json file
        # data = {
        #     "requests": self.requests
        # }
        # filename = 'data.json'
        # file_path = os.path.join(os.getcwd(), filename)
        # with open(filename, 'w') as json_file:
        #     json.dump(data, json_file, indent=4)
        #
        # print(f"JSON file '{filename}' created successfully at {file_path}")
        self.polygon_list = [
            Polygon([(0, 0), (0, self.cake_len), (self.cake_width, self.cake_len), (self.cake_width, 0)])]

        if self.use_gui:
            self.canvas = tk.Canvas(self.root, height=self.canvas_height, width=self.canvas_width, bg="#FCF1E3")
            self.canvas.pack()
            self.draw_cake()
            self.root.mainloop()
        else:
            self.play_game()

    def generate_requests(self):
        while 1:
            requests_sum = 0
            self.requests = []
            numRequests = self.rng.integers(10, 101)

            for _ in range(numRequests - 1):
                # Generate a float number between 10 and 100
                request = self.rng.uniform(10, 100)
                self.requests.append(request)
                requests_sum += request

            if self.validate_requests():
                break
            else:
                print("Invalid requests generated. Regenerating...")

    def validate_requests(self):
        # Check the sum of requests should be less than cake size.
        print("Requests: ", self.requests)
        print("Sum of requests: ", np.sum(self.requests))
        if np.sum(self.requests) > 10000:
            return False

        # Check the number of requests should be between 1 and 100
        if len(self.requests) < 1 or len(self.requests) > 100:
            return False

        # Check each request should be between 10 and 100
        for request in self.requests:
            if request < 10 or request > 100:
                return False

        # Find shape of a cake with length l and 1.6l to fit all requests
        self.cake_len = round(math.sqrt(1.05 * np.sum(self.requests) / 1.6), 2)
        self.cake_width = round(self.cake_len * 1.6, 2)

        self.scale = 700/self.cake_len
        print("Cake size: ", self.cake_len * self.cake_width)
        return True

    def resume(self):
        if self.game_state == "pause":
            self.game_state = "resume"
            self.game_speed = "normal"
            self.root.after(50, self.play_game)

    def pause(self):
        if self.game_state != "over":
            self.game_state = "pause"

    def step(self):
        if self.game_state != "over":
            self.game_state = "pause"
            self.root.after(100, self.play_game)

    def toggle_speed(self):
        if self.game_state == "resume":
            if self.game_speed == "normal":
                self.game_speed = "fast"
            else:
                self.game_speed = "normal"

    def divide_polygon(self, polygon, line):
        """
        Divide a convex polygon by a line segment into two polygons.

        Parameters:
        - polygon: A convex polygon (as a Shapely Polygon object)
        - line_points: A list containing two points that represent the line segment

        Returns:
        - Two polygons (as shapely Polygon objects) that result from dividing the original polygon
        """
        # Create the convex polygon and the line segment using Shapely
        # polygon = Polygon(polygon_points)
        # line = LineString(line_points)

        # make polygon input geometry valid
        if self.player_name == "Group 2":
            polygon = polygon.convex_hull

        # Check if the line intersects with the polygon
        if not line.intersects(polygon):
            return [polygon]
        # Split the polygon into two pieces
        result = split(polygon, line)

        # Convert the result into individual polygons

        polygons = []
        for i in range(len(result.geoms)):
            polygons.append(result.geoms[i])

        return polygons

    def play_game(self):
        self.turns += 1

        # Create the state object for the player
        before_state = PieceOfCakeState(self.polygon_list, self.cur_pos, self.turns, self.requests, self.cake_len, self.cake_width)
        returned_action = None
        if (not self.player_timeout) and self.timeout_warning_count < 3:
            player_start = time.time()
            try:
                # Call the player's move function for turn on this move
                returned_action = self.player.move(
                    current_percept=before_state
                )
            except Exception as e:
                print(f"Exception in player code: {e}")
                traceback.print_exc()
                returned_action = None

            player_time_taken = time.time() - player_start
            self.logger.debug("Player {} took {:.3f}s".format(self.player_name, player_time_taken))
            if player_time_taken > 10:
                self.logger.warning("Player {} took {:.3f}s".format(self.player_name, player_time_taken))
                self.timeout_warning_count += 1

            self.player_time -= player_time_taken
            if self.player_time <= 0:
                self.player_timeout = True
                returned_action = None

        print("Move received: ", returned_action)
        if self.check_action(returned_action):
            if self.check_and_apply_action(returned_action):
                print("Move Accepted! New position", self.cur_pos)
                self.logger.debug("Received move from {}".format(self.player_name))
                self.valid_moves += 1
            else:
                self.logger.info("Invalid move from {} as it does not follow the rules".format(self.player_name))
        else:
            print("Invalid move")
            self.logger.info("Invalid move from {} as it doesn't follow the return format".format(self.player_name))

        if self.use_gui:
            self.draw_cake()


        print("Turn {} complete".format(self.turns))

        if self.penalty is not None:
            self.game_state = "over"
            print("Assignment completed!\n\n Total Penalty: {}\n".format(self.penalty))
            self.end_time = time.time()
            print("\nTime taken: {}\n".format(self.end_time - self.start_time))
            return

        if self.turns < self.max_turns and not self.player_timeout:
            if self.use_gui:
                if self.game_state == "resume":
                    if self.game_speed == "normal":
                        self.root.after(200, self.play_game)
                    else:
                        self.root.after(5, self.play_game)
            else:
                self.play_game()
        else:
            print("Timeout: Pieces not assigned...\n\n")
            self.game_state = "over"
            self.end_time = time.time()
            print("\nTime taken: {}\nValid moves: {}\n".format(self.end_time - self.start_time, self.valid_moves))
            return

    # Verify the action returned by the player
    def check_action(self, action):
        print(action)
        print("Checking action: ", action)
        if action is None:
            print("No action returned")
            return False
        if type(action) is not tuple:
            return False
        if len(action) != 2:
            return False
        if type(action[0]) is not int:
            return False
        if action[0] < 1 or action[0] > 3:
            return False
        if type(action[1]) is not list:
            return False
        if action[0] == constants.INIT and self.turns != 1:
            return False
        if (action[0] == constants.INIT or action[0] == constants.CUT) and len(action[1]) != 2:
            return False
        # Check if action[1] is a list of float values with maximum 2 decimal places
        if (action[0] == constants.INIT or action[0] == constants.CUT) and not all(isinstance(x, (int, float)) and x == round(x, 2) for x in action[1]):
            return False

        # For assign action, check
        # 1 if the length of the list is equal to the number of requests
        # 2 All the values which are greater than -1 are unique
        # 3 All the values are integers and greater than -1

        if action[0] == constants.ASSIGN:
            if len(action[1]) != len(self.requests):
                return False
            temp = [x for x in action[1] if x != -1]
            if len(set(temp)) != len(temp):
                return False
            if not all(isinstance(x, int) and x >= -1 for x in action[1]):
                return False

        return True

    def invalid_knife_position(self, pos):
        cur_x, cur_y = pos
        if (cur_x != 0 and cur_x != self.cake_width) and (cur_y != 0 and cur_y != self.cake_len):
            return True

        if cur_x == 0 or cur_x == self.cake_width:
            if cur_y < 0 or cur_y > self.cake_len:
                return True

        if cur_y == 0 or cur_y == self.cake_len:
            if cur_x < 0 or cur_x > self.cake_width:
                return True
        return False

    def check_and_apply_action(self, action):
        if action[0] == constants.INIT:
            if self.invalid_knife_position(action[1]):
                return False
            self.cur_pos = action[1]
            return True
        elif action[0] == constants.CUT:
            cur_x, cur_y = action[1]

            # Check if the next position is on the boundary of the cake
            if self.invalid_knife_position(action[1]):
                return False
            
            # If the next position is same then the cut is invalid
            if self.cur_pos[0] == cur_x and self.cur_pos[1] == cur_y:
                return False

            # If the cut has already been made then it's invalid
            if self.prev_pos is not None and ((self.prev_pos[0], self.prev_pos[1], cur_x, cur_y) in self.cake_cuts or (cur_x, cur_y, self.prev_pos[0], self.prev_pos[1]) in self.cake_cuts):
                return False

            # Check if the cut is horizontal across the cake boundary
            if cur_x == 0 or cur_x == self.cake_width:
                if self.cur_pos[0] == cur_x:
                    return False

            # Check if the cut is vertical across the cake boundary
            if cur_y == 0 or cur_y == self.cake_len:
                if self.cur_pos[1] == cur_y:
                    return False

            # Cut the cake piece
            newPieces = []
            for polygon in self.polygon_list:
                line_points = LineString([tuple(self.cur_pos), tuple(action[1])])
                slices = self.divide_polygon(polygon, line_points)
                for slice in slices:
                    newPieces.append(slice)

            self.polygon_list = newPieces
            self.prev_pos = self.cur_pos
            self.cur_pos = action[1]
            return True
        elif action[0] == constants.ASSIGN:
            self.penalty = 0
            self.assignment = action[1]
            for request_index, assignment in enumerate(action[1]):
                # check if the cake piece fit on a plate of diameter 25 and calculate penaly accordingly
                if assignment == -1 or (not self.can_cake_fit_in_plate(self.polygon_list[assignment])):
                    self.penalty += 100
                else:
                    penalty_percentage = 100 * abs(self.polygon_list[assignment].area - self.requests[request_index])/self.requests[request_index]
                    if penalty_percentage > self.tolerance:
                        self.penalty += penalty_percentage
            self.prev_pos = None
            return True
        return False

    def centroid(self, polygon):
        """
        Calculate the centroid (geometric center) of a polygon.

        Parameters:
        - polygon_points: List of tuples representing vertices of the polygon (in order)

        Returns:
        - The centroid as a tuple (x, y)
        """
        centroid_point = polygon.centroid
        return (centroid_point.x, centroid_point.y)

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        - point1, point2: Tuples representing two points (x1, y1), (x2, y2)

        Returns:
        - The Euclidean distance between the points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def can_cake_fit_in_plate(self, cake_piece, radius=12.5):
        """
        Check if the cake can fit inside a plate of radius 12.5.

        Parameters:
        - cake_pieces: Cake pieces (as shapely Polygon object)
        - radius: The radius of the circle

        Returns:
        - True if the cake can fit inside the plate, False otherwise
        """
        # Return True if cake piece has de minimis area
        if cake_piece.area < 0.25:
            return True

        # Step 1: Get the points on the cake piece and store as numpy array
        cake_points = np.array(list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double)

        # Step 2: Find the minimum bounding circle of the cake piece
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def draw_cake(self):
        self.canvas.delete("position")
        # Create a rectangle for the background of length and width l and 1.6l
        self.canvas.create_rectangle(self.x_offset, self.y_offset, self.x_offset + self.cake_width * self.scale, self.y_offset + self.cake_len * self.scale, tags="background", fill="#FFD1DC")

        self.canvas.create_text(self.scale*self.cake_width + 100, 50, text="Request ID", font=("Arial", 14), fill="black",
                                activefill="gray", tags="requests")

        self.canvas.create_text(self.scale * self.cake_width + 200, 50, text="Requests", font=("Arial", 14),
                                fill="black",
                                activefill="gray", tags="requests")

        if self.assignment is not None:
            self.canvas.create_text(self.scale * self.cake_width + 300, 50, text="Assignment", font=("Arial", 14),
                                fill="black",
                                activefill="gray", tags="requests")

        # Next to the cake, print all the requests
        for i, request in enumerate(self.requests):
            x = self.scale*self.cake_width + 200
            y = 50 + (i+1) * 20
            # print the request along with the position
            self.canvas.create_text(x-100, y, text="{}".format(i), font=("Arial", 14), fill="black",
                                    activefill="gray", tags="requests")
            self.canvas.create_text(x, y, text="{:.2f}".format(request), font=("Arial", 14), fill="black",
                                    activefill="gray", tags="requests")

            if self.assignment is not None:
                if self.can_cake_fit_in_plate(self.polygon_list[self.assignment[i]]):
                    self.canvas.create_text(x+100, y, text="{}".format(round(self.polygon_list[self.assignment[i]].area, 2)), font=("Arial", 14), fill="green",
                                        activefill="gray", tags="requests")
                else:
                    self.canvas.create_text(x + 100, y,
                                            text="{}".format(round(self.polygon_list[self.assignment[i]].area, 2)),
                                            font=("Arial", 14), fill="red",
                                            activefill="gray", tags="requests")

        # Mark the start, cur, and end positions
        if self.cur_pos is not None:
            self.mark_position(self.cur_pos)
        # self.mark_position(self.end_pos, "red")
        self.create_buttons()
        if self.prev_pos is not None:
            self.cake_cuts.append((self.prev_pos[0], self.prev_pos[1],
                                    self.cur_pos[0], self.cur_pos[1]))

        for cut in self.cake_cuts:
            self.canvas.create_line(self.x_offset + cut[0]*self.scale, self.y_offset + cut[1]*self.scale, self.x_offset + cut[2]*self.scale, self.y_offset + cut[3]*self.scale, fill="black", width=2, tags="cuts")

        # Get centroid of each polygon and mark it's area
        for polygon in self.polygon_list:
            cent = self.centroid(polygon)
            if self.can_cake_fit_in_plate(polygon):
                self.mark_area(cent, "green", polygon.area)
            else:
                self.mark_area(cent, "red", polygon.area)

        if self.penalty is not None:
            self.canvas.create_text(700, 20, text="Penalty: {}".format(self.penalty), font=("Arial", 14), fill="black",
                                              activefill="gray", tags="penalty text")
            # Calculate the sum of lengths of all the cuts
            total_length = 0
            for cut in self.cake_cuts:
                total_length += self.euclidean_distance((cut[0], cut[1]), (cut[2], cut[3]))
            self.canvas.create_text(900, 20, text="Total Length: {:.2f}".format(total_length), font=("Arial", 14), fill="black",
                                              activefill="gray", tags="penalty text")

    def create_buttons(self):
        # Create text-based "Pause" button on the canvas
        self.pause_btn = self.canvas.create_text(250, 20, text="Pause", font=("Arial", 14), fill="black",
                                                 activefill="gray", tags="pause_button")
        self.canvas.tag_bind("pause_button", "<Button-1>", lambda e: self.pause())

        # Create a text-based "Reset" button on the canvas
        self.resume_btn = self.canvas.create_text(350, 20, text="Start/Resume", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="resume_button")
        self.canvas.tag_bind("resume_button", "<Button-1>", lambda e: self.resume())

        self.resume_btn = self.canvas.create_text(450, 20, text="1X/4X", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="speed_button")
        self.canvas.tag_bind("speed_button", "<Button-1>", lambda e: self.toggle_speed())

        self.step_btn = self.canvas.create_text(550, 20, text="Step", font=("Arial", 14), fill="black",
                                                  activefill="gray", tags="step_button")
        self.canvas.tag_bind("step_button", "<Button-1>", lambda e: self.step())

    def mark_area(self, pos, color, area):
        x, y = pos

        x1, y1 = self.x_offset + x * self.scale, self.y_offset + y * self.scale
        self.canvas.create_text(x1, y1, text="{:.2f}".format(area), font=("Arial", 15), fill=color, tags="position", )

    def mark_position(self, pos):
        x, y = pos

        x1, y1 = self.x_offset + x * self.scale, self.y_offset + y * self.scale
        r = self.scale/8
        self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, fill="", outline="blue", width=1, tags="position")
