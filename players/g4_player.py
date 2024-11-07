import os
import pickle
from typing import List

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

import constants
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import copy
from tqdm import tqdm


class Player:
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
        self.cuts = None

    def move(self, current_percept) -> (int, List[int]):
        """Function which retrieves the current state of the cake

        Args:
            current_percept(PieceOfCakeState): contains current state information
        Returns:
            (int, List[int]): This function returns the next move of the user:
            The integer return value should be one of the following:
                constants.INIT - If wants to initialize the knife position
                constants.CUT - If wants to cut the cake
                constants.ASSIGN - If wants to assign the pieces
        """
        turn_number = current_percept.turn_number
        requests = current_percept.requests
        polygons = current_percept.polygons

        if turn_number == 1:
            cake_len = current_percept.cake_len
            cake_width = current_percept.cake_width

            num_cuts = len(requests)
            num_restarts = 10
            num_iterations = 100

            min_loss = float("inf")

            for restart in range(num_restarts):
                cuts = generate_random_cuts(num_cuts, (cake_width, cake_len))
                loss = self.get_loss_from_cuts(cuts, current_percept)
                print(f"Restart {restart} Loss: {loss}")

                if loss < min_loss:
                    best_cuts = copy.deepcopy(cuts)
                    min_loss = loss

                cuts = best_cuts

                # Gradient descent
                learning_rate = 0.1

                epoch = 0
                while loss > 0.01 and epoch < num_iterations:
                    gradients = self.get_gradient(loss, cuts, current_percept)

                    cur_x, cur_y = cuts[0]
                    for j in range(len(cuts)):
                        cuts[j] = get_shifted_cut(
                            cuts[j],
                            -learning_rate * gradients[j],
                            (cake_width, cake_len),
                            (cur_x, cur_y),
                        )
                        cur_x, cur_y = cuts[j]
                    loss = self.get_loss_from_cuts(cuts, current_percept)
                    if loss < min_loss:
                        best_cuts = copy.deepcopy(cuts)
                        min_loss = loss
                    print(f"Iteration: {epoch}, Loss: {loss}")
                    epoch += 1

            print(f"Best penalty: {min_loss * 100}")

            self.cuts = [[round(cut[0], 2), round(cut[1], 2)] for cut in best_cuts]
            return constants.INIT, self.cuts[0]
        elif turn_number <= len(self.cuts):
            return constants.CUT, self.cuts[turn_number - 1]

        return constants.ASSIGN, optimal_assignment(
            requests, [polygon.area for polygon in polygons]
        )

    def get_loss_from_cuts(self, cuts, current_percept):
        new_percept = copy.deepcopy(current_percept)
        new_polygons = new_percept.polygons

        new_percept.cur_pos = cuts[0]

        for cut in cuts[1:]:
            new_polygons, new_percept = self.cut_cake(
                cut,
                new_polygons,
                new_percept,
            )
        loss = cost_function(new_polygons, current_percept.requests)

        return loss

    def get_gradient(self, loss, cuts, current_percept):
        dw = current_percept.cake_width / 100
        gradients = np.zeros(len(cuts))

        cur_x, cur_y = cuts[0]
        for i in range(len(cuts)):
            # for i in tqdm(range(len(cuts))):
            new_cuts = copy.deepcopy(cuts)
            new_cuts[i] = get_shifted_cut(
                cuts[i],
                dw,
                (current_percept.cake_width, current_percept.cake_len),
                (cur_x, cur_y),
            )
            cur_x, cur_y = new_cuts[i]

            new_loss = self.get_loss_from_cuts(new_cuts, current_percept)
            gradients[i] = (new_loss - loss) / dw
        return gradients

    def cut_cake(self, cut, polygons, current_percept):
        # Check if the next position is on the boundary of the cake
        if invalid_knife_position(cut, current_percept):
            raise ValueError("Invalid knife position")

        # Cut the cake piece
        newPieces = []
        for polygon in polygons:
            line_points = LineString([tuple(current_percept.cur_pos), tuple(cut)])
            slices = divide_polygon(polygon, line_points)
            for slice in slices:
                newPieces.append(slice)

        current_percept.cur_pos = cut
        return newPieces, current_percept


def invalid_knife_position(pos, current_percept):
    cur_x, cur_y = pos
    if (cur_x != 0 and cur_x != current_percept.cake_width) and (
        cur_y != 0 and cur_y != current_percept.cake_len
    ):
        return True

    if cur_x == 0 or cur_x == current_percept.cake_width:
        if cur_y < 0 or cur_y > current_percept.cake_len:
            return True

    if cur_y == 0 or cur_y == current_percept.cake_len:
        if cur_x < 0 or cur_x > current_percept.cake_width:
            return True
    return False


def divide_polygon(polygon, line):
    if not line.intersects(polygon):
        return [polygon]
    result = split(polygon, line)

    polygons = []
    for i in range(len(result.geoms)):
        polygons.append(result.geoms[i])

    return polygons


def generate_random_cuts(num_cuts, cake_dims):
    cake_width, cake_len = cake_dims
    corner_gap = 1e-3
    cur_x, cur_y = [
        [0, np.random.uniform(corner_gap, cake_len - corner_gap)],
        [cake_width, np.random.uniform(corner_gap, cake_len - corner_gap)],
        [np.random.uniform(corner_gap, cake_width - corner_gap), 0],
        [np.random.uniform(corner_gap, cake_width - corner_gap), cake_len],
    ][np.random.choice(4)]

    cuts = [[cur_x, cur_y]]
    for i in range(num_cuts):
        # Generate random cuts
        top = [
            np.random.uniform(corner_gap, cake_width - corner_gap),
            0,
        ]
        bottom = [
            np.random.uniform(corner_gap, cake_width - corner_gap),
            cake_len,
        ]
        right = [
            cake_width,
            np.random.uniform(corner_gap, cake_len - corner_gap),
        ]
        left = [
            0,
            np.random.uniform(corner_gap, cake_len - corner_gap),
        ]

        if cur_x == 0:  # Start from left
            cuts.append([top, bottom, right][np.random.choice(3)])
        elif cur_x == cake_width:  # Start from right
            cuts.append([top, bottom, left][np.random.choice(3)])
        elif cur_y == 0:  # Start from top
            cuts.append([bottom, left, right][np.random.choice(3)])
        elif cur_y == cake_len:  # Start from bottom
            cuts.append([top, left, right][np.random.choice(3)])

        cur_x, cur_y = cuts[-1]

    return cuts


def cost_function(polygons, requests):
    R = requests
    V = [polygon.area for polygon in polygons]

    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Calculate the total cost by summing the optimal assignment costs
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return total_cost


def get_shifted_cut(cut, shift, cake_dims, pos):
    cake_width, cake_len = cake_dims
    cur_x, cur_y = pos

    shifted_cut = copy.deepcopy(cut)
    if cut[0] == 0:  # Left
        if cut[1] + shift > cake_len:
            if cur_y != cake_len:
                remainder = (
                    cut[1] + shift - cake_len
                    if cut[1] + shift - cake_len < cake_width
                    else cake_width
                )
                shifted_cut = [remainder, cake_len]
        elif cut[1] + shift < 0:
            if cur_y != 0:
                remainder = (
                    -(cut[1] + shift) if -(cut[1] + shift) < cake_width else cake_width
                )
                shifted_cut = [remainder, 0]
        else:
            shifted_cut = [0, cut[1] + shift]
    elif cut[0] == cake_width:  # Right
        if cut[1] + shift > cake_len:
            if cur_y != cake_len:
                remainder = cut[1] - shift if cake_width - (cut[1] - shift) > 0 else 0
                shifted_cut = [cake_width - remainder, cake_len]
        elif cut[1] + shift < 0:
            if cur_y != 0:
                remainder = (
                    -(cut[1] + shift) if cake_width - (cut[1] + shift) > 0 else 0
                )
                shifted_cut = [cake_width - remainder, 0]
        else:
            shifted_cut = [cake_width, cut[1] + shift]
    elif cut[1] == 0:  # Top
        if cut[0] + shift > cake_width:
            if cur_x != cake_width:
                remainder = (
                    cut[0] + shift - cake_width
                    if cut[0] + shift - cake_width < cake_len
                    else cake_len
                )
                shifted_cut = [cake_width, remainder]
        elif cut[0] + shift < 0:
            if cur_x != 0:
                remainder = (
                    -(cut[0] + shift) if -(cut[0] + shift) < cake_len else cake_len
                )
                shifted_cut = [0, remainder]
        else:
            shifted_cut = [cut[0] + shift, 0]
    elif cut[1] == cake_len:  # Bottom
        if cut[0] + shift > cake_width:
            if cur_x != cake_width:
                remainder = cut[0] - shift if cake_len - (cut[0] - shift) > 0 else 0
                shifted_cut = [
                    cake_width,
                    cake_len - remainder,
                ]
        elif cut[0] + shift < 0:
            if cur_x != 0:
                remainder = -(cut[0] + shift) if cake_len - (cut[0] + shift) > 0 else 0
                shifted_cut = [0, cake_len - remainder]
        else:
            shifted_cut = [cut[0] + shift, cake_len]
    return shifted_cut


def optimal_assignment(R, V):
    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Assignment array where assignment[i] is the index of V matched to R[i]
    assignment = [int(col_indices[i]) for i in range(num_requests)]

    return assignment
