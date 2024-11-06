import logging
import os
import pickle
from collections import deque
from typing import List, Tuple
from enum import Enum

import miniball
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from shapely.ops import split
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, rand, anneal
import optuna

import constants

class SearchMethod(Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'
    LOCAL = 'local'

def assign_polygons_to_requests(polygons, requests, tolerance):
    num_requests = len(requests)
    num_polygons = len(polygons)

    cost_matrix = np.full((num_requests, num_polygons), fill_value=1e6)

    for i, request_area in enumerate(requests):
        for j, polygon in enumerate(polygons):
            piece_area = polygon.area

            size_difference = abs(piece_area - request_area)
            percentage_difference = (size_difference / request_area) * 100
            
            if percentage_difference <= tolerance:
                area_penalty = 0
            else:
                area_penalty = percentage_difference
                
            fits_on_plate = can_cake_fit_in_plate(polygon)
            plate_penalty = 0 if fits_on_plate else 1000

            total_penalty = area_penalty + plate_penalty

            cost_matrix[i, j] = total_penalty

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_penalty = cost_matrix[row_ind, col_ind].sum()

    assignment = [-1] * num_requests
    assigned_polygons = set()

    for req_idx, poly_idx in zip(row_ind, col_ind):
        penalty = cost_matrix[req_idx, poly_idx]
        if penalty < 1e5:
            assignment[req_idx] = int(poly_idx)
            assigned_polygons.add(poly_idx)
        else:
            pass

    return total_penalty, assignment
    
def can_cake_fit_in_plate(cake_piece, radius=12.5):
    cake_points = np.array(list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double)
    res = miniball.miniball(cake_points)
    return res["radius"] <= radius

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
        
        self.num_splits = 1
        self.horizontal_split_gap = 24.6
        self.triangle_viable = 23.507
        self.vertical_split_gap = 4
        self.preplanned_moves = deque()
        self.request_served = 0
        
        self.current_percept = None

    def move(self, current_percept) -> Tuple[int, List[int]]:
        self.current_percept = current_percept
        if current_percept.cake_len <= self.triangle_viable:
            return self.triangle()
        return self.quadrangle()

    def init_cuts(self, current_percept) -> Tuple[List[List[int]]]:
        cake_len, cake_width = current_percept.cake_len, current_percept.cake_width

        working_pairs = []
        extra_pieces = 0
        extra_area = cake_len * cake_width * 0.05
        while not working_pairs:
            mod_requests = current_percept.requests
            mod_requests += [extra_area / extra_pieces for _ in range(extra_pieces)]
            factor_pairs = self.find_factor_pairs(len(mod_requests))

            for factor_pair in factor_pairs:
                num_rows, num_cols = factor_pair
                row_height, col_width = cake_len / num_rows, cake_width / num_cols
                fits_plate = np.sqrt(row_height**2 + col_width**2) <= 25
                if fits_plate:
                    working_pairs.append(factor_pair)

            extra_pieces += 1

        cuts = {}
        for working_pair in working_pairs:
            num_rows, num_cols = working_pair
            row_height, col_width = cake_len / num_rows, cake_width / num_cols
            horizontal_cuts = [[i * row_height, i * row_height] for i in range(1, num_rows)]
            vertical_cuts = [[i * col_width, i * col_width] for i in range(1, num_cols)]
            cuts[working_pair] = (horizontal_cuts, vertical_cuts)

        return cuts

    def quadrangle(self):
        polygons = self.current_percept.polygons
        turn_number = self.current_percept.turn_number
        cur_pos = self.current_percept.cur_pos
        requests = self.current_percept.requests
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width
        #added
        num_requests = len(requests)

        # if turn_number == 1:
        #     self.grid_optimizer = GridOptimizer(
        #         cake_width=cake_width,
        #         cake_len=cake_len,
        #         num_row_divisions=4,
        #         num_col_divisions=5,
        #         requests=requests,
        #         tolerance=self.tolerance,
        #         learning_rate=0.01,
        #         convergence_threshold=1e-4
        #     )
        #     self.grid_optimizer.run_optimization(max_evals=1000, method=SearchMethod.UNIFORM)
            
        #     return constants.INIT, [0, 0.01]
        
        #New fix:
        if turn_number == 1:
            # Ensure that the grid can handle all requests
            min_num_rows = int(np.ceil(np.sqrt(num_requests)))
            # min_num_cols = int(np.ceil(num_requests * min_num_rows/num_requests))
            min_num_cols = int(np.ceil(num_requests / min_num_rows))

            if min_num_rows * min_num_cols < num_requests:
                min_num_rows += 1  
                min_num_cols = int(np.ceil(num_requests / min_num_rows))

            self.grid_optimizer = GridOptimizer(
                cake_width=cake_width,
                cake_len=cake_len,
                num_row_divisions=min_num_rows, 
                num_col_divisions=min_num_cols, 
                requests=requests,
                tolerance=self.tolerance,
                learning_rate=0.01,
                convergence_threshold=1e-4
            )
            self.grid_optimizer.run_optimization(max_evals=1000, method=SearchMethod.UNIFORM)
            
            return constants.INIT, [0, 0.01]
        
        
        if turn_number == 2:
            cur_x, cur_y = cur_pos[0], cur_pos[1]
            horizontal_cuts = self.grid_optimizer.horizontal_cuts
            vertical_cuts = self.grid_optimizer.vertical_cuts
            
            for cut in horizontal_cuts:
                if cur_x == 0:
                    self.shift_along((cur_x, cur_y), cut[0])
                    cur_x, cur_y = cut[1][0], cut[1][1]
                    self.preplanned_moves.append([cur_x, cur_y])
                else:
                    self.shift_along((cur_x, cur_y), cut[1])
                    cur_x, cur_y = cut[0][0], cut[0][1]
                    self.preplanned_moves.append([cur_x, cur_y])
            
            if cur_x == cake_width:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
                self.preplanned_moves.append([0, cake_len - 0.01])
            self.preplanned_moves.append([0.01, cake_len])
            cur_x, cur_y = 0.01, cake_len
            
            for cut in vertical_cuts:
                if cur_y == cake_len:
                    self.shift_along((cur_x, cur_y), cut[1])
                    cur_x, cur_y = cut[0][0], cut[0][1]
                    self.preplanned_moves.append([cur_x, cur_y])
                else:
                    self.shift_along((cur_x, cur_y), cut[0])
                    cur_x, cur_y = cut[1][0], cut[1][1]
                    self.preplanned_moves.append([cur_x, cur_y])

        if self.preplanned_moves:
            dest_x, dest_y = self.preplanned_moves.popleft()
            return constants.CUT, [round(dest_x, 2), round(dest_y, 2)]
        
        _, assignment = assign_polygons_to_requests(polygons, requests, self.tolerance) 
        return constants.ASSIGN, assignment

    def shift_along(self, cur_pos, target_pos):
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width
        
        if cur_pos[0] == 0:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([0.01, 0])
            else:
                self.preplanned_moves.append([0.01, cake_len])
        elif cur_pos[0] == cake_width:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([cake_width - 0.01, 0])
            else:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
        elif cur_pos[1] == 0:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, 0.01])
            else:
                self.preplanned_moves.append([cake_width, 0.01])
        else:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, cake_len - 0.01])
            else:
                self.preplanned_moves.append([cake_width, cake_len - 0.01])
        self.preplanned_moves.append([target_pos[0], target_pos[1]])

    def triangle(self):
        turn_number = self.current_percept.turn_number
        cur_pos = self.current_percept.cur_pos
        requests = sorted(self.current_percept.requests)
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width

        if turn_number == 1:
            self.preplanned_moves.append([0,0])
            return constants.INIT, [0,0]
        
        if self.request_served < len(requests):
            area = requests[self.request_served]
            base = round(2 * area / cake_len, 2)

            cur_x, cur_y = cur_pos[0], cur_pos[1]

            if turn_number == 2:
                if cake_len ** 2 + base ** 2 > 25 ** 2:
                    raise Exception("First cut doesn't fit on plate.")
                dest_x, dest_y = base, cake_len
            else:
                dest_x = round(self.preplanned_moves[-2][0] + base, 2)
                dest_y = cake_len if cur_y == 0 else 0

                if dest_x > cake_width:
                    l1 = dest_x - cake_width
                    l2 = cake_width - self.preplanned_moves[-1][0]
                    h1 = (cake_len * (l1)) / l2
                    h2 = cake_len - h1
                    h3 = (h1 * (l1)) / l2
                    new_y = round(h2 - h3, 2)
                    dest_y = new_y if cur_y == 0 else round(cake_len - new_y, 2)
                    dest_x = cake_width

            self.preplanned_moves.append([dest_x, dest_y])
            self.request_served += 1
            return constants.CUT, [dest_x, dest_y]
        
        _, assignment = assign_polygons_to_requests(self.current_percept.polygons, self.current_percept.requests, self.tolerance) 
        return constants.ASSIGN, assignment

class GridOptimizer:
    def __init__(self, cake_width, cake_len, num_row_divisions, num_col_divisions, requests, tolerance=5,
                 learning_rate=0.01, max_iterations=100, convergence_threshold=1e-4):
        """
        Initialize the GridOptimizer with cake dimensions, grid parameters, and optimization settings.

        Parameters:
        - cake_width: float, width of the cake.
        - cake_len: float, length of the cake.
        - num_row_divisions: int, number of desired rows (vertical divisions).
        - num_col_divisions: int, number of desired columns (horizontal divisions).
        - requests: List[float], list of requested areas.
        - tolerance: float, tolerance percentage for area differences.
        - learning_rate: float, step size for gradient descent updates.
        - max_iterations: int, maximum number of iterations for gradient descent.
        - convergence_threshold: float, threshold for convergence based on gradient norm.
        """
        self.cake_width = cake_width
        self.cake_len = cake_len
        self.num_row_divisions = num_row_divisions  # Desired number of rows
        self.num_col_divisions = num_col_divisions  # Desired number of columns
        self.num_horizontal_cuts = self.num_row_divisions - 1  # Number of horizontal cuts
        self.num_vertical_cuts = self.num_col_divisions - 1   # Number of vertical cuts

        self.requests = requests
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold

        self.horizontal_cuts = []  # Each cut is [(x1, y1), (x2, y2)]
        self.vertical_cuts = []

        y_positions = np.linspace(0, self.cake_len, self.num_row_divisions + 1)[1:-1]
        for y in y_positions:
            self.horizontal_cuts.append([(0, y), (self.cake_width, y)])

        x_positions = np.linspace(0, self.cake_width, self.num_col_divisions + 1)[1:-1]
        for x in x_positions:
            self.vertical_cuts.append([(x, 0), (x, self.cake_len)])
        
        self.polygons = []

    def generate_polygons(self):
        """
        Generate polygons formed by the cuts defined by lines with endpoints on opposite edges.

        Returns:
        - polygons: List[Polygon], list of Shapely Polygon objects representing the cake pieces.
        """
        cake_polygon = Polygon([
            (0, 0),
            (self.cake_width, 0),
            (self.cake_width, self.cake_len),
            (0, self.cake_len)
        ])

        polygons = [cake_polygon]

        for cut in self.vertical_cuts:
            line = LineString(cut)
            new_polygons = []
            for poly in polygons:
                new_pieces = self.divide_polygon(poly, line)
                new_polygons.extend(new_pieces)
            polygons = new_polygons

        for cut in self.horizontal_cuts:
            line = LineString(cut)
            new_polygons = []
            for poly in polygons:
                new_pieces = self.divide_polygon(poly, line)
                new_polygons.extend(new_pieces)
            polygons = new_polygons

        self.polygons = polygons
        return polygons

    def divide_polygon(self, polygon, line):
        """
        Divide a polygon by a line into two or more polygons.

        Parameters:
        - polygon: A Shapely Polygon object.
        - line: A LineString object representing the cutting line.

        Returns:
        - List of Shapely Polygon objects resulting from the division.
        """
        if not line.intersects(polygon):
            return [polygon]

        result = split(polygon, line)
        return list(result.geoms)

    def cost_function(self):
        """
        Compute the total penalty using the assignment function.

        Returns:
        - total_penalty: float, the total penalty calculated from the assignment.
        """
        polygons = self.generate_polygons()
        total_penalty, _ = assign_polygons_to_requests(
            polygons=polygons,
            requests=self.requests,
            tolerance=self.tolerance
        )
        if len(polygons) > self.num_col_divisions * self.num_row_divisions:
            total_penalty += 100000
        return total_penalty

    def objective(self, params):
        """
        Objective function for optimization. Updates the cut positions based on sampled parameters.

        Parameters:
        - params: dict, parameters sampled by Hyperopt.

        Returns:
        - total_penalty: float, the total penalty calculated from the assignment.
        """
        for key in params:
            if 'h_y' in key:
                params[key] = max(0, min(self.cake_len, params[key]))
            elif 'v_x' in key:
                params[key] = max(0, min(self.cake_width, params[key]))

        self.horizontal_cuts = [
            [(0, params[f'h_y1_{i}']), (self.cake_width, params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]

        self.vertical_cuts = [
            [(params[f'v_x1_{i}'], 0), (params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]

        return self.cost_function()
    
    def run_optimization(self, max_evals=1000, range_x=0.1, range_y=0.1, method=SearchMethod.UNIFORM):
        if method == SearchMethod.UNIFORM:
            self.run_optimization_uniform(max_evals, range_x, range_y)
        elif method == SearchMethod.NORMAL:
            self.run_optimization_normal(max_evals, range_x, range_y)
        elif method == SearchMethod.LOCAL:
            self.run_local_search(max_evals)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def run_optimization_uniform(self, max_evals=1000, range_x=0.1, range_y=0.1):
        """
        Run the optimization using Optuna.

        Updates the cut positions to minimize the total penalty.
        """
        h_y_positions = np.linspace(0, self.cake_len, self.num_row_divisions + 1)[1:-1]
        v_x_positions = np.linspace(0, self.cake_width, self.num_col_divisions + 1)[1:-1]

        delta_y = self.cake_len * range_y
        delta_x = self.cake_width * range_x

        def objective(trial):
            params = {
                **{f'h_y1_{i}': trial.suggest_uniform(f'h_y1_{i}', max(0, y - delta_y), min(self.cake_len, y + delta_y))
                for i, y in enumerate(h_y_positions)},
                **{f'h_y2_{i}': trial.suggest_uniform(f'h_y2_{i}', max(0, y - delta_y), min(self.cake_len, y + delta_y))
                for i, y in enumerate(h_y_positions)},
                **{f'v_x1_{i}': trial.suggest_uniform(f'v_x1_{i}', max(0, x - delta_x), min(self.cake_width, x + delta_x))
                for i, x in enumerate(v_x_positions)},
                **{f'v_x2_{i}': trial.suggest_uniform(f'v_x2_{i}', max(0, x - delta_x), min(self.cake_width, x + delta_x))
                for i, x in enumerate(v_x_positions)}
            }

            return self.objective(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max_evals)

        best_params = study.best_params

        self.horizontal_cuts = [
            [(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]

        self.vertical_cuts = [
            [(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]

        self.generate_polygons()
        self.display_polygons(self.polygons)

        print("Best horizontal lines:")
        for i in range(self.num_horizontal_cuts):
            print([(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])])

        print("Best vertical lines:")
        for i in range(self.num_vertical_cuts):
            print([(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)])
    
    # def run_optimization_uniform(self, max_evals=1000, range_x=0.1, range_y=0.1):
    #     """
    #     Run the optimization using Hyperopt's Tree-structured Parzen Estimator (TPE).

    #     Updates the cut positions to minimize the total penalty.
    #     """
    #     h_y_positions = np.linspace(0, self.cake_len, self.num_row_divisions + 1)[1:-1]
    #     v_x_positions = np.linspace(0, self.cake_width, self.num_col_divisions + 1)[1:-1]

    #     delta_y = self.cake_len * range_x 
    #     delta_x = self.cake_width * range_y

    #     space = {
    #         **{f'h_y1_{i}': hp.uniform(f'h_y1_{i}',
    #                                     max(0, y - delta_y),
    #                                     min(self.cake_len, y + delta_y))
    #         for i, y in enumerate(h_y_positions)},
    #         **{f'h_y2_{i}': hp.uniform(f'h_y2_{i}',
    #                                     max(0, y - delta_y),
    #                                     min(self.cake_len, y + delta_y))
    #         for i, y in enumerate(h_y_positions)},
    #         **{f'v_x1_{i}': hp.uniform(f'v_x1_{i}',
    #                                     max(0, x - delta_x),
    #                                     min(self.cake_width, x + delta_x))
    #         for i, x in enumerate(v_x_positions)},
    #         **{f'v_x2_{i}': hp.uniform(f'v_x2_{i}',
    #                                     max(0, x - delta_x),
    #                                     min(self.cake_width, x + delta_x))
    #         for i, x in enumerate(v_x_positions)},
    #     }

    #     best_params = fmin(
    #         fn=self.objective,
    #         space=space,
    #         algo=tpe.suggest,
    #         max_evals=max_evals
    #     )

    #     self.horizontal_cuts = [
    #         [(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])]
    #         for i in range(self.num_horizontal_cuts)
    #     ]

    #     self.vertical_cuts = [
    #         [(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)]
    #         for i in range(self.num_vertical_cuts)
    #     ]

    #     self.generate_polygons()
    #     self.display_polygons(self.polygons)

    #     print("Best horizontal lines:")
    #     for i in range(self.num_horizontal_cuts):
    #         print([
    #             (0, best_params[f'h_y1_{i}']), 
    #             (self.cake_width, best_params[f'h_y2_{i}'])
    #         ])

    #     print("Best vertical lines:")
    #     for i in range(self.num_vertical_cuts):
    #         print([
    #             (best_params[f'v_x1_{i}'], 0), 
    #             (best_params[f'v_x2_{i}'], self.cake_len)
    #         ])
    
    def run_optimization_normal(self, max_evals=1000, range_x=0.005, range_y=0.005):
        """
        Run the optimization using Hyperopt's Tree-structured Parzen Estimator (TPE).

        Updates the cut positions to minimize the total penalty.
        """
        h_y_positions = np.linspace(0, self.cake_len, self.num_row_divisions + 1)[1:-1]
        v_x_positions = np.linspace(0, self.cake_width, self.num_col_divisions + 1)[1:-1]

        delta_y = self.cake_len * range_y
        delta_x = self.cake_width * range_x

        space = {
            **{f'h_y1_{i}': hp.normal(f'h_y1_{i}', y, delta_y)
            for i, y in enumerate(h_y_positions)},
            **{f'h_y2_{i}': hp.normal(f'h_y2_{i}', y, delta_y)
            for i, y in enumerate(h_y_positions)},
            **{f'v_x1_{i}': hp.normal(f'v_x1_{i}', x, delta_x)
            for i, x in enumerate(v_x_positions)},
            **{f'v_x2_{i}': hp.normal(f'v_x2_{i}', x, delta_x)
            for i, x in enumerate(v_x_positions)},
        }

        best_params = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals
        )

        self.horizontal_cuts = [
            [(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]

        self.vertical_cuts = [
            [(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]

        self.generate_polygons()
        self.display_polygons(self.polygons)

        print("Best horizontal lines:")
        for i in range(self.num_horizontal_cuts):
            print([
                (0, best_params[f'h_y1_{i}']),
                (self.cake_width, best_params[f'h_y2_{i}'])
            ])

        print("Best vertical lines:")
        for i in range(self.num_vertical_cuts):
            print([
                (best_params[f'v_x1_{i}'], 0),
                (best_params[f'v_x2_{i}'], self.cake_len)
            ])
    
    def run_local_search(self, max_evals=1000):
        delta = 0.1
        # Initial parameters
        h_y_positions = np.linspace(0, self.cake_len, self.num_row_divisions + 1)[1:-1]
        v_x_positions = np.linspace(0, self.cake_width, self.num_col_divisions + 1)[1:-1]

        # Initialize parameters and track best configuration
        params = {}
        for i, y in enumerate(h_y_positions):
            params[f'h_y1_{i}'] = y
            params[f'h_y2_{i}'] = y

        for i, x in enumerate(v_x_positions):
            params[f'v_x1_{i}'] = x
            params[f'v_x2_{i}'] = x

        # Set initial best loss and parameters
        best_loss = self.objective(params)
        best_params = params.copy()

        for iter_num in range(max_evals):
            print(f"Iteration {iter_num + 1}/{max_evals}, Current best loss: {best_loss}")
            improved = False  # Track if any improvement happened in this iteration

            # Test each parameter by increasing or decreasing it
            for key in params:
                original_value = params[key]

                # Try increasing the parameter by delta
                params[key] = min(original_value + delta, self.cake_len if 'h_y' in key else self.cake_width)
                loss = self.objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()
                    improved = True
                else:
                    # Try decreasing the parameter by delta if no improvement was found
                    params[key] = max(original_value - delta, 0)
                    loss = self.objective(params)
                    if loss < best_loss:
                        best_loss = loss
                        best_params = params.copy()
                        improved = True
                    else:
                        # Revert to original value if neither adjustment improved
                        params[key] = original_value

            # # Stop if no improvement was found during this iteration
            if not improved:
                print("No improvement found in this iteration. Stopping search.")
                break

        # Set cuts to the best parameters found
        self.horizontal_cuts = [
            [(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]
        self.vertical_cuts = [
            [(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]

        # Generate polygons and display
        self.generate_polygons()
        self.display_polygons(self.polygons)

        # Output the best cut configuration
        print("Best horizontal lines:")
        for i in range(self.num_horizontal_cuts):
            print([(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])])

        print("Best vertical lines:")
        for i in range(self.num_vertical_cuts):
            print([(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)])

    def display_polygons(self, polygons):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        
        for poly in polygons:
            if isinstance(poly, Polygon):
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, edgecolor='black')
                
                area = poly.area
                
                centroid = poly.centroid
                ax.text(centroid.x, centroid.y, f"{area:.2f}", ha='center', va='center', fontsize=8, color='blue')
        
        ax.set_title("Polygon Grid Visualization")
        plt.xlabel("Width")
        plt.ylabel("Length")
        plt.show()