from typing import List, Callable
import numpy as np
import logging
import constants
import miniball
from enum import Enum
from shapely.geometry import Polygon

from piece_of_cake_state import PieceOfCakeState
from players.g2.helpers import *
from players.g2.even_cuts import *
from players.g2.uneven_cuts import *
from players.g2.best_combination import best_combo, cuts_to_moves
from players.g2.assigns import (
    assign,
    sorted_assign,
    hungarian_min_penalty,
    dp_min_penalty,
    greedy_best_fit_assignment,
)


class Strategy(Enum):
    EVEN = "even"
    UNEVEN = "uneven"
    CLIMB_HILLS = "climb_hills"
    SAWTOOTH = "sawtooth"
    BEST_CUTS = "best_cuts"


class G2_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        tolerance: int,
    ) -> None:
        """Initialise the player with the basic information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            precomp_dir (str): Directory path to store/load pre-computation
            tolerance (int): tolerance for the cake distribution
            cake_len (int): Length of the smaller side of the cake
        """

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        # stores next actions in a queue
        # each action is a ({INIT | CUT | ASSIGN}, list) tuple
        self.move_queue: list[tuple[int, list]] = []
        self.strategy = None
        self.requestscut = 0
        self.move_object = None

    def cut(self, cake_len, cake_width, cur_pos) -> tuple[int, List[int]]:
        if cur_pos[0] == 0:
            return constants.CUT, [cake_width, round((cur_pos[1] + 5) % cake_len, 2)]
        else:
            return constants.CUT, [0, round((cur_pos[1] + 5) % cake_len, 2)]

    def assign(
        self, assign_func: Callable[[list[Polygon], list[float]], list[int]]
    ) -> tuple[int, List[int]]:

        assignment: list[int] = assign_func(
            self.polygons, self.requests, self.tolerance
        )

        return constants.ASSIGN, assignment

    def can_cake_fit_in_plate(self, cake_piece: Polygon, radius=12.5):
        if cake_piece.area < 0.25:
            return True

        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def __calculate_penalty(
        self, assign_func: Callable[[list[Polygon], list[float]], list[int]]
    ) -> float:
        penalty = 0
        assignments: list[int] = assign_func(self.polygons, self.requests)

        for request_index, assignment in enumerate(assignments):
            # check if the cake piece fit on a plate of diameter 25 and calculate penaly accordingly
            if assignment == -1 or (
                not self.can_cake_fit_in_plate(self.polygons[assignment])
            ):
                penalty += 100
            else:
                penalty_percentage = (
                    100
                    * abs(self.polygons[assignment].area - self.requests[request_index])
                    / self.requests[request_index]
                )
                if penalty_percentage > self.tolerance:
                    penalty += penalty_percentage
        return penalty

    def climb_hills(self):
        current_penalty = self.__calculate_penalty(assign)
        print(f"1 penalty: {current_penalty}")
        current_penalty = self.__calculate_penalty(assign)
        print(f"2 penalty: {current_penalty}")

        if self.turn_number == 1:
            print()
            return constants.INIT, [0, 0]

        if len(self.polygons) < len(self.requests):
            if self.cur_pos[0] == 0:
                return constants.CUT, [
                    self.cake_width,
                    round((self.cur_pos[1] + 5) % self.cake_len, 2),
                ]
            else:
                return constants.CUT, [
                    0,
                    round((self.cur_pos[1] + 5) % self.cake_len, 2),
                ]

        return constants.ASSIGN, assign(self.polygons, self.requests)

    def best_cuts(self):
        # initialize move queue
        if not self.move_queue:
            if self.turn_number != 1:
                print(f"assigning now!")
                return self.assign(greedy_best_fit_assignment)

            print(f"I'll think for a while now..")
            best_cuts = best_combo(
                self.requests, self.cake_len, self.cake_width, self.tolerance
            )

            self.move_queue = cuts_to_moves(
                best_cuts, self.requests, self.cake_len, self.cake_width
            )

        # get the next move from the move queue
        return self.move_queue.pop(0)

    def process_percept(self, current_percept: PieceOfCakeState):
        self.polygons = current_percept.polygons
        self.turn_number = current_percept.turn_number
        self.cur_pos = current_percept.cur_pos
        self.requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width
        self.cake_area = self.cake_len * self.cake_width
        self.requestlength = len(self.requests)

    def decide_strategy(self):
        if is_uniform(self.requests, self.tolerance):
            self.strategy = Strategy.EVEN
            self.move_object = EvenCuts(self.requests, self.cake_width, self.cake_len)
        elif grid_enough(self.requests, self.cake_width, self.cake_len, self.tolerance):
            self.strategy = Strategy.UNEVEN
            self.move_object = UnevenCuts(self.requests, self.cake_width, self.cake_len)
        else:  # Default
            self.strategy = Strategy.BEST_CUTS
            self.move_object = UnevenCuts(self.requests, self.cake_width, self.cake_len)

    def move(self, current_percept: PieceOfCakeState) -> tuple[int, List[int]]:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement"""
        self.process_percept(current_percept)
        if self.turn_number == 1:
            self.decide_strategy()

        # for only 1 request:
        if self.requestlength == 1:
            if self.turn_number == 1:
                return constants.INIT, [0, 0]
            elif self.requestscut < self.requestlength:
                self.requestscut += 1
                return constants.CUT, [
                    round(2 * 0.05 * self.requests[0] / self.cake_len, 2),
                    self.cake_len,
                ]

            return self.assign(assign)

        elif self.cake_area <= 860:
            if self.cake_len <= 23.507:
                if self.turn_number == 1:
                    self.move_queue.append([0, 0])
                    return constants.INIT, [0, 0]

                if self.requestscut < self.requestlength:
                    polygonarea = self.requests[self.requestscut]
                    polygonbase = round(2 * polygonarea / self.cake_len, 2)

                    if self.cur_pos[1] == 0:
                        # we're on top side of the board
                        if self.turn_number == 2:
                            next_move = [polygonbase, self.cake_len]
                        else:
                            x = round(self.move_queue[-2][0] + polygonbase, 2)
                            y = self.cake_len

                            if x > self.cake_width:
                                x = self.cake_width
                                y = round(
                                    2
                                    * self.cake_area
                                    * 0.05
                                    / (self.cake_width - self.move_queue[-2][0]),
                                    2,
                                )
                            next_move = [x, y]

                        self.move_queue.append(next_move)
                        self.requestscut += 1
                        return constants.CUT, next_move
                    else:
                        # we're on bottom of the board
                        x = round(self.move_queue[-2][0] + polygonbase, 2)
                        y = 0

                        if x > self.cake_width:
                            x = self.cake_width
                            y = self.cake_len - round(
                                2
                                * self.cake_area
                                * 0.05
                                / (self.cake_width - self.move_queue[-2][0]),
                                2,
                            )
                        next_move = [x, y]
                        self.move_queue.append(next_move)
                        self.requestscut += 1
                        return constants.CUT, next_move
                return self.assign(assign)

        elif self.strategy == Strategy.EVEN:
            move = self.move_object.move(self.turn_number, self.cur_pos)
            if move == None:
                if len(self.requests) < 10:
                    # print("Brute Force!")
                    return self.assign(greedy_best_fit_assignment)
                else:
                    return self.assign(greedy_best_fit_assignment)
                return self.assign(assign)

            return move

        elif self.strategy == Strategy.UNEVEN:
            move = self.move_object.move(self.turn_number, self.cur_pos)
            if move == None:
                return self.assign(assign)

            return move

        elif self.strategy == Strategy.CLIMB_HILLS:
            return self.climb_hills()
        elif self.strategy == Strategy.BEST_CUTS:
            return self.best_cuts()

        # default
        return self.climb_hills()
