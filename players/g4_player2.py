import os
import pickle
from typing import List

import numpy as np
import logging

import constants
from scipy.optimize import linear_sum_assignment

EASY_LEN_BOUND = 23.5
ANGLE = 50
INIT_POS_RATIO = 0.6

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, tolerance: int) -> None:
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
        self.knife_pos = []  # Persistent knife position for continuity
        self.cut_count = 0

    def calc_start(self, requests):
        area = sum(requests) * 1.05
        h = np.sqrt(area / 1.6)
        return [0, round(h * INIT_POS_RATIO, 2)]
    
    def distance_to_boundary(self, cur_pos, cake_width, cake_len):
        x, y = cur_pos
        dx = (cake_width - x) if np.cos(self.angle) > 0 else x
        dy = (cake_len - y) if np.sin(self.angle) > 0 else y
        return min(dx / abs(np.cos(self.angle)), dy / abs(np.sin(self.angle)))
    
    def hungarian_method(self, R, V):
        num_requests = len(R)
        num_values = len(V)

        cost_matrix = np.zeros((num_requests, num_values))

        for i, r in enumerate(R):
            for j, v in enumerate(V):
                cost_matrix[i][j] = abs(r - v) / r

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
        assignment = [int(col_indices[i]) for i in range(num_requests)]

        return assignment

    def move(self, current_percept) -> (int, List[int]):
        """Perform valid cuts based on area-based strategy and assign pieces."""
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        requests = sorted(current_percept.requests)  # Sorted in ascending order
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cur_pos = current_percept.cur_pos
        num_requests = len(requests)
        cake_area = cake_len * cake_width

        # zig-zag method
        if cake_len <= EASY_LEN_BOUND:
            if turn_number == 1:
                self.knife_pos.append([0, 0])
                return constants.INIT, [0, 0]
            
            if self.cut_count < num_requests:
                curr_polygon_area = requests[self.cut_count]
                curr_polygon_base = round(2 * curr_polygon_area / cake_len, 2)

                if cur_pos[1] == 0: # knife is currently on the top cake edge
                    if turn_number == 2:
                        next_knife_pos = [curr_polygon_base, cake_len]
                    else:
                        next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                        next_y = cake_len
                        # When knife goes over the cake width
                        if next_x > cake_width:
                            next_x = cake_width
                            next_y = round(2 * cake_area * 0.05 / (cake_width - self.knife_pos[-2][0]), 2)
                        next_knife_pos = [next_x, next_y]

                    self.knife_pos.append(next_knife_pos)
                    self.cut_count += 1
                    return constants.CUT, next_knife_pos
                
                else: # knife is currently on the bottom cake edge
                    next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                    next_y = 0
                    # when knife goes over the cake width
                    if next_x > cake_width:
                        next_x = cake_width
                        next_y = cake_len - round(2 * cake_area * 0.05 / (cake_width - self.knife_pos[-2][0]), 2)
                    next_knife_pos = [next_x, next_y]
                    
                    self.knife_pos.append(next_knife_pos)
                    self.cut_count += 1
                    return constants.CUT, next_knife_pos
                
        else:
            # Rhombus Cutting at 50 degrees
            if turn_number == 1:
                self.angle = np.radians(ANGLE)
                starting_pos = self.calc_start(requests)
                self.knife_pos.append(starting_pos)

                return constants.INIT, starting_pos
                
            if self.cut_count < num_requests:
                x, y = cur_pos
                dist = self.distance_to_boundary(cur_pos, cake_width, cake_len)
                x += np.cos(self.angle) * dist
                y += np.sin(self.angle) * dist

                # Check for boundary collisions and adjust angle
                if x <= 0 or x >= cake_width: 
                    self.angle = np.pi - self.angle
                elif y <= 0 or y >= cake_len:
                    self.angle = -self.angle

                next_knife_pos = [round(x, 2), round(y, 2)]
                self.knife_pos.append(next_knife_pos)
                self.cut_count += 1

                return constants.CUT, next_knife_pos


        V = [p.area for p in polygons]
        assignment = self.hungarian_method(requests, V)

        return constants.ASSIGN, assignment
     

    # def zig_zag():


    # def move(self, current_percept) -> (int, List[int]):
    #     """Perform valid cuts based on area-based strategy and assign pieces."""
    #     polygons = current_percept.polygons
    #     turn_number = current_percept.turn_number
    #     requests = sorted(current_percept.requests)  # Sorted in ascending order
    #     cake_len = current_percept.cake_len
    #     cake_width = current_percept.cake_width

    #     # First turn initializes knife position at the top-left corner
    #     if turn_number == 1:
    #         self.knife_pos = [0, 0]
    #         return constants.INIT, [0, 0]

    #     # Cutting Strategy
    #     if len(polygons) < len(requests):
    #         # Start by dividing the cake in the shorter dimension (width or length)
    #         if turn_number == 2:
    #             if cake_len < cake_width:
    #                 cut_position = cake_len / 2
    #                 self.knife_pos = [0, cut_position]
    #                 return constants.CUT, [cake_width, cut_position]  # Horizontal cut
    #             else:
    #                 cut_position = cake_width / 2
    #                 self.knife_pos = [cut_position, 0]
    #                 return constants.CUT, [cut_position, cake_len]  # Vertical cut

    #         # For subsequent cuts, alternate between width and length divisions
    #         next_request_area = requests[len(polygons)]
    #         if self.knife_pos[1] == cake_len:  # Current position is at top or bottom
    #             cut_x = round(self.knife_pos[0] + next_request_area / cake_len, 2)
    #             self.knife_pos = [cut_x, 0] if self.knife_pos[1] == cake_len else [cut_x, cake_len]
    #             return constants.CUT, [cut_x, self.knife_pos[1]]
    #         else:  # Current position is at left or right side
    #             cut_y = round(self.knife_pos[1] + next_request_area / cake_width, 2)
    #             self.knife_pos = [0, cut_y] if self.knife_pos[0] == cake_width else [cake_width, cut_y]
    #             return constants.CUT, [self.knife_pos[0], cut_y]

    #     # Assignment Strategy
    #     assignment = self._assign_requests(polygons, requests)
    #     return constants.ASSIGN, assignment

    def _assign_requests(self, polygons, requests) -> List[int]:
        """Assign each polygon piece to the closest-sized request to minimize penalty."""
        assignments = []
        polygon_areas = [p.area for p in polygons]
        
        # Sort polygons and requests by area in ascending order
        polygon_indices = list(np.argsort(polygon_areas))
        request_indices = list(np.argsort(np.argsort(requests)))

        # Assign each request to a polygon based on area matching
        for request_idx in request_indices:
            assignments.append(polygon_indices[request_idx])

        return assignments
