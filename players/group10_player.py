from collections import defaultdict
import os
import pickle
from typing import List

import numpy as np
import math
import logging

import constants


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
        self.cake_len = None
        self.cake_width = None
        self.cake_diagonal = None
        self.requests = None
        self.cuts = []
        self.base_case_switch = False
        self.working_height = None
        self.uniform_mode = False
        self.uniform_cuts = []
        self.used_crumb = [[0.00, 0.00], [0.00, 0.00], [0.00, 0.00], [0.00, 0.00]] #LEFT-TOP, RIGHT-TOP, LEFT-BOTTOM, RIGHT-BOTTOM. The values inside is [x, y]

    def move(self, current_percept) -> tuple[int, List[int]]:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

            Args:
                current_percept(TimingMazeState): contains current state information
            Returns:
                int: This function returns the next move of the user:
                    WAIT = -1
                    LEFT = 0
                    UP = 1
                    RIGHT = 2
                    DOWN = 3
        """
        extra_tol = 40 #Change if we suck

        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cut_number = current_percept.turn_number - 1
        cur_pos = current_percept.cur_pos
    
        if turn_number == 1:
            # initialize instance variables, sorted requests
            self.requests = current_percept.requests
            self.cake_len = current_percept.cake_len
            self.cake_width = current_percept.cake_width
            self.cake_diagonal = self.calcDiagonal()
            print ("cake_len:", self.cake_len)
            print ("cake_width:", self.cake_width)
            print ("cake_diagonal:", self.cake_diagonal)
            if not self.uniform_mode:
                # print("BEFORE")
                is_uniform, grid_area = self.if_uniform(current_percept.requests, extra_tol)
                # print("AFTER")
                print(is_uniform, grid_area)
                if is_uniform:
                    self.grid_cut(current_percept, grid_area)
                    self.uniform_mode = True

        # TODO: adjust for the case when the base the triangle needs surpasses the cake width we have (first occurence, switch to the length, after this switch, make sure we continue working on that edge). CURRENT PROGRAM UNABLE TO MOVE IF WE SWITCH SIDES FROM THE BOTTOM EDGE
        # case if the diagonal of the total cake is <= 25
        if self.cake_diagonal <= 25:
            if turn_number == 1:
                self.cuts.append((0, 0))
                return constants.INIT, [0,0]

            # assign pieces
            if (cut_number > len(self.requests)):
                # print("I AM TRYING TO ASSIGN PIECES")
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            
            current_area = self.requests[cut_number - 1]
            
            x = cur_pos[0]
            y = cur_pos[1]

            if self.base_case_switch:
                # TODO: this currently retraces the past cut and creates triangles from the same point
                x = self.cuts[cut_number - 2][0]
                y = self.cuts[cut_number - 2][1]
                if y != 0:
                    y = self.cuts[cut_number - 2][1] - round(2 * current_area / self.working_height, 2)
                
                self.cuts.append((x, y))
                return constants.CUT, [x, y]

            if (cut_number == 1):
                x = round(2 * current_area / self.cake_len, 2)
            else:
                x = round(self.cuts[cut_number - 2][0] + (2 * current_area / self.cake_len), 2)

            y = (0, self.cake_len) [cut_number % 2]

            if x > self.cake_width:
                area_left = (self.cake_len * self.cake_width) / 1.05 * .05 # finding the extra cake portion
                self.working_height = self.cake_width - cur_pos[0]
                if (cut_number < len(self.requests)): # not on our last request
                    area_left += sum(self.requests[cut_number:])
                x = self.cake_width
                y = round(2 * area_left / self.working_height, 2)
                self.base_case_switch = True

            self.cuts.append((x, y))
            return constants.CUT, [x, y]
        elif self.uniform_mode:
            if (cut_number > len(self.uniform_cuts) - 1):
                # print("I AM TRYING TO ASSIGN PIECES")
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            # print("I AM IN UNIFORM MODE")
            # print("Turn Number: ", turn_number)
            # print("Cut Number: ", cut_number)
            if turn_number == 1:
                # print(self.uniform_cuts)
                # print(self.uniform_cuts[cut_number])
                return constants.INIT, self.uniform_cuts[cut_number]

            return constants.CUT, self.uniform_cuts[cut_number]
        else:
            if len(polygons) != len(self.requests):
                if cur_pos[0] == 0:
                    return constants.CUT, [self.cake_width, round((cur_pos[1] + 5)%self.cake_len, 2)]
                else:
                    return constants.CUT, [0, round((cur_pos[1] + 5)%self.cake_len, 2)]

            assignment = []
            for i in range(len(self.requests)):
                assignment.append(i)

            return constants.ASSIGN, assignment
    
    def calcDiagonal(self):
        return (math.sqrt((self.cake_len * self.cake_len) + (self.cake_width * self.cake_width)))
    
    def assignPolygons(self, polygons) -> list[int]:
        # parse polygons to polygon_areas: dict(rank: (area, i))
        # print(polygons)
        polygon_areas = []
        requests_items = []
        for i in range(len(polygons)):
            polygon_areas.append((polygons[i].area, i)) 
        for i in range(len(self.requests)):
            requests_items.append((self.requests[i], i))
        # print("AREAS: ", polygon_areas)
        # print("REQUESTS: ", requests_items)

        matches = {} # request : polygon

        for i in range(len(requests_items)):
            # print(f"I AM ON {i} ITERATION")
            min_diff = math.inf
            polygon = -1
            polygon_index = -1
            # print("INIT COMPLETE")
            for j in range(len(polygon_areas)):
                # print("area ", polygon_areas[j][0])
                # print("request ", requests_items[i][0])
                temp_diff = abs(float(polygon_areas[j][0]) - float(requests_items[i][0]))
                # print("TEMP DIFF SET")
                if temp_diff < min_diff:
                    min_diff = temp_diff
                    polygon = polygon_areas[j][1]
                    polygon_index = j
            # print ("polygon:", polygon)
            matches[i] = polygon
            polygon_areas.pop(polygon_index)
        
        assignment = []
        for i in range(len(requests_items)):
            if i in matches:
                assignment.append(matches[i])
            else:
                assignment.append(-1)
        print("THIS IS MY ASSIGNMENT", assignment)
        return assignment
                
    #Returns whether if it is uniform, and the area of the uniform. If not, area is -1
    def if_uniform(self, requests, extra_tol=0.0) -> tuple[bool, float]:
        tolerance = (self.tolerance + extra_tol) / 100
        skewed_average = (1 - tolerance**2)*(sum(requests) / len(requests)) if tolerance < 1 else (tolerance**2)**(sum(requests) / len(requests))
        print("THIS IS OUR TOLERANCE: ", tolerance)
        print("THIS IS OUR TRUE AVG: ", sum(requests)/len(requests))
        print("THIS IS OUR SKEWED AVG: ", skewed_average)
        for req in requests:
            # print("WE ARE CHECKING THIS REQ: ", req)
            # print("WE ARE CHECKING IF: ", abs(req-skewed_average)/req)
            # print("GREATER THAN: ", tolerance)
            if (abs(req-skewed_average)/req) >= tolerance:
                return (False, -1)
            
        return (True, skewed_average)
    
    #Cuts into grid using crumbs method, returns the coordinate to cut to.
    def grid_cut(self, current_percept, grid_area) -> list[float, float]:
        # print("IN GRID CUT")
        print(self.cake_len)
        # length_shave = 0.01266028 * self.cake_len
        
        # width_shave = round(1.6 * length_shave, 2)
        # length_shave = round(length_shave, 2)

        width_shave = 0.01
        length_shave = 0.01

        # print("BEFORE FACTORS")
        s_factor, l_factor = self.find_closest_factors()

        print("MY FACTORS: ", s_factor, l_factor)
        x_cuts = [width_shave] # first cut
        y_cuts = [length_shave] # first cut
        x_increment = round((self.cake_width - 2*width_shave)/l_factor, 2)
        y_increment = round((self.cake_len - 2*length_shave)/s_factor, 2)
        # x_increment * y_increment should be equal to grid_area;
        # but do we need to check it? They might be some lil decimal points off tho;
        # but does it matter?
        for ind in range(s_factor):
            y_cuts.append(round(y_cuts[ind] + y_increment, 2))
        for ind in range(l_factor):
            x_cuts.append(round(x_cuts[ind] + x_increment, 2))

        print("X CUTS: ", x_cuts)
        print("Y CUTS: ", y_cuts)

        for i in range(len(y_cuts)):
            if i == 0:
                self.uniform_cuts.append([0, y_cuts[i]])
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])
            elif i % 2 == 1:
                self.uniform_cuts.append([round(self.cake_width - (0.01 + self.used_crumb[1][0]), 2), 0])
                # self.used_crumb[1][0] += 0.01
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])
                self.uniform_cuts.append([0, y_cuts[i]])
            elif i % 2 == 0:
                self.uniform_cuts.append([round(0.01 + self.used_crumb[0][0], 2), 0])
                # self.used_crumb[0][0] += 0.01
                self.uniform_cuts.append([0, y_cuts[i]])
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])

        for i in range(len(x_cuts)):
            if l_factor % 2 == 1:
                if i == 0:
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                    self.uniform_cuts.append([x_cuts[i], 0])
                elif i % 2 == 1:
                    self.uniform_cuts.append([0, round((0.01 + self.used_crumb[0][1]), 2)])
                    # self.used_crumb[0][1] += 0.01
                    self.uniform_cuts.append([x_cuts[i], 0])
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                elif i % 2 == 0:
                    self.uniform_cuts.append([0, round(self.cake_len - (0.01 + self.used_crumb[2][1]), 2)])
                    # self.used_crumb[2][1] += 0.01
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                    self.uniform_cuts.append([x_cuts[i], 0])
            elif l_factor % 2 == 0:
                if i == 0:
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                elif i % 2 == 1:
                    self.uniform_cuts.append([self.cake_width, round((0.01 + self.used_crumb[1][1]), 2)])
                    # self.used_crumb[1][1] += 0.01
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                elif i % 2 == 0:
                    self.uniform_cuts.append([self.cake_width, round(self.cake_len - (0.01 + self.used_crumb[3][1]), 2)])
                    # self.used_crumb[3][1] += 0.01
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                
           
        print("PATH TO VICTORY: ", self.uniform_cuts)
        
    #Finds the closest factor with the number of requests
    def find_closest_factors(self) -> list[int, int]: # [smaller_closest_factor, bigger_closest_factor]
        num_of_requests = len(self.requests)
        factors = []
        for num in range(int(num_of_requests**0.5),0,-1):
            if num_of_requests % num == 0:
                factors.append([num, num_of_requests//num])
            if factors:
                return factors[0]
        # TODO: Here we can add a check to see if num_of_requests is prime, by checking if factors[-1][0] == 1
        
        




        
        
