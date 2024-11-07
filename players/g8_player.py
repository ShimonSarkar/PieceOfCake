import os
import pickle
from typing import List

import logging
from uuid import uuid4
import numpy as np

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from scipy.optimize import linear_sum_assignment

import miniball
from tqdm import tqdm

import constants
from piece_of_cake_state import PieceOfCakeState


class G8_Player:
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

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        self.requests = None
        self.cut_path = []
        self.assignment = []
        self.inital_state = None
        self.edges = None
        self.cake = None
        self.solution = None
        self.beam_uuid_to_pieces = {}

    def move(self, current_percept: PieceOfCakeState) -> tuple[int, List[int]]:
        """Function that retrieves the current state of the cake map and returns an cake movement

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
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        # On turn 1 we need to decide where to start.
        if turn_number == 1:
            # initialize these variables
            self.assignment = [-1 for _ in range(len(requests))]
            self.requests = requests
            self.cake = polygons[0]
            self.cake_len = cake_len
            self.cake_width = cake_width

            self.edges = [
                ((0, 0), (cake_width, 0)),  # top
                ((cake_width, 0), (cake_width, cake_len)),  # right
                ((cake_width, cake_len), (0, cake_len)),  # bottom
                ((0, cake_len), (0, 0)),  # left
            ]

            self.solution = self.solve()
            x, y = self.solution.pop(0)

            return constants.INIT, [round(x, 2), round(y, 2)]

        if len(self.solution) == 0:
            return self.assign_polygons(current_percept.polygons)

        x, y = self.solution.pop(0)

        return constants.CUT, [round(x, 2), round(y, 2)]

    def solve(self) -> list[tuple[float, float]]:
        """Find optimal cutting sequence using beam search"""
        beam_width = min(50, len(self.requests) * 5)
        max_depth = len(self.requests) + 1
        counter = 0

        # Initialize beam with possible first points
        initial_points = self.generate_initial_points()
        beam = [(0, [point], uuid4()) for point in initial_points]

        best_solution = None
        best_score = float("inf")

        with tqdm(total=max_depth) as pbar:
            while counter < max_depth:
                new_beam = []

                for _, current_points, uuid in beam:
                    next_points = self.generate_next_points(current_points[-1])

                    for next_point in next_points:
                        if len(current_points) > 1:
                            if not self.is_valid_cut(
                                current_points[-1], next_point, current_points
                            ):
                                continue

                        new_points = current_points + [next_point]
                        penalty, cut_length, new_beam_uuid = self.evaluate_cut_sequence(
                            new_points, uuid
                        )
                        score = penalty + cut_length * 1e-6

                        if penalty > best_score:
                            continue

                        if penalty <= best_score:
                            best_score = penalty
                            best_solution = new_points

                        new_beam.append((score, new_points, new_beam_uuid))

                    # We should never use this beam uuids again
                    if uuid in self.beam_uuid_to_pieces:
                        del self.beam_uuid_to_pieces[uuid]

                # Keep best beam_width solutions
                beam = sorted(new_beam, key=lambda x: x[0])[:beam_width]
                counter += 1
                pbar.update(1)

            if best_solution is None:
                raise ValueError("No valid solution found")

        return best_solution

    def evaluate_cut_sequence(
        self, points: list[tuple[float, float]], uuid
    ) -> tuple[float, float]:
        """Evaluate a sequence of cut endpoints, returning (penalty, total_cut_length)"""
        # Convert points to cuts
        cuts = []
        for i in range(len(points) - 1):
            cuts.append(LineString([points[i], points[i + 1]]))

        cut = cuts[0] if len(cuts) == 1 else cuts[-1]

        pieces = [self.cake] if len(cuts) == 1 else self.beam_uuid_to_pieces[uuid]

        new_pieces = []
        for piece in pieces:
            if cut.intersects(piece):
                split_result = split(piece, cut)
                new_pieces.extend(list(split_result.geoms))
            else:
                new_pieces.append(piece)
        pieces = new_pieces

        new_beam_uuid = uuid4()
        self.beam_uuid_to_pieces[new_beam_uuid] = pieces

        # Calculate penalties (handles both area mismatches and plate fitting)
        penalties = self.calculate_penalties(pieces)
        return penalties, self.calculate_cut_length(points), new_beam_uuid

    def generate_initial_points(self):
        """Generate possible starting points along the perimeter"""
        points = []
        samples = 5

        for (x1, y1), (x2, y2) in self.edges:
            for t in np.linspace(0, 1, samples):
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((x, y))

        return points

    def generate_next_points(self, current_point: tuple[float, float]):
        """Generate possible next points on valid edges"""
        points = []
        samples = 12

        current_edge = self.get_edge(current_point)

        for i, ((x1, y1), (x2, y2)) in enumerate(self.edges):
            if i != current_edge:  # Can't cut to same edge
                for t in np.linspace(0, 1, samples):
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    points.append((x, y))

        return points

    def calculate_cut_length(self, points: list[tuple[float, float]]) -> float:
        """Calculate total length of cut sequence"""
        total_length = 0

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return total_length

    def get_edge(self, point: tuple[int, int], tolerance: float = 1e-10):
        """Get what edge a point lies on"""

        x, y = point

        if abs(y - 0) <= tolerance:
            return 0  # top

        if abs(x - self.cake_width) <= tolerance:
            return 1  # right

        if abs(y - self.cake_len) <= tolerance:
            return 2  # bottom

        if abs(x - 0) <= tolerance:
            return 3  # left

        raise ValueError(f"Point {point} not on any edge")

    def fits_on_plate(self, cake_piece, radius=12.5):
        """
        Check if the cake can fit inside a plate of radius 12.5.

        Parameters:
        - cake_pieces: Cake pieces (as shapely Polygon object)
        - radius: The radius of the circle

        Returns:
        - True if the cake can fit inside the plate, False otherwise
        """
        # Step 1: Get the points on the cake piece and store as numpy array

        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )

        # Step 2: Find the minimum bounding circle of the cake piece
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def calculate_penalties(self, pieces: list[Polygon]):
        """Calculate penalties exactly as the game does"""
        n_pieces = len(pieces)
        n_requests = len(self.requests)
        max_size = max(n_pieces, n_requests)

        # Create cost matrix
        cost_matrix = np.full((max_size, max_size), 100.0)

        # Fill in penalties exactly as game calculates them
        for i, piece in enumerate(pieces):
            if not self.fits_on_plate(piece):
                continue

            area = piece.area
            for j, request in enumerate(self.requests):
                # Calculate exactly as game does
                percentage_diff = 100 * abs(area - request) / request
                if percentage_diff <= self.tolerance:
                    cost_matrix[i, j] = 0  # No penalty within tolerance
                else:
                    cost_matrix[i, j] = percentage_diff  # Full percentage as penalty

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Calculate total penalty exactly as game would
        total_penalty = 0
        assignments = [-1] * n_requests

        # Make assignments and calculate penalties
        for piece_idx, request_idx in zip(row_ind, col_ind):
            if piece_idx < n_pieces and request_idx < n_requests:
                assignments[request_idx] = piece_idx

        # Calculate penalty exactly as game does
        for request_idx, piece_idx in enumerate(assignments):
            if piece_idx == -1 or not self.fits_on_plate(pieces[piece_idx]):
                total_penalty += 100
            else:
                percentage_diff = (
                    100
                    * abs(pieces[piece_idx].area - self.requests[request_idx])
                    / self.requests[request_idx]
                )
                if percentage_diff > self.tolerance:
                    total_penalty += percentage_diff

        return total_penalty

    def assign_polygons(self, pieces: list[Polygon]) -> tuple[str, list[int]]:
        """
        Get optimal mapping of requests to pieces using exact game penalty calculation.
        """
        n_pieces = len(pieces)
        n_requests = len(self.requests)
        max_size = max(n_pieces, n_requests)

        # Create cost matrix
        cost_matrix = np.full(
            (max_size, max_size), 100.0
        )  # Default to 100 for invalid assignments

        # Fill in costs exactly as game calculates them
        for i, piece in enumerate(pieces):
            if not self.fits_on_plate(piece):
                continue

            area = piece.area
            for j, request in enumerate(self.requests):
                # Calculate percentage deviation exactly as game does
                penalty_percentage = 100 * abs(area - request) / request

                # Only add penalty if it exceeds tolerance
                if penalty_percentage > self.tolerance:
                    cost_matrix[i, j] = penalty_percentage
                else:
                    cost_matrix[i, j] = 0  # No penalty if within tolerance

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create assignments list
        assignments = [-1] * n_requests
        for piece_idx, request_idx in zip(row_ind, col_ind):
            if piece_idx < n_pieces and request_idx < n_requests:
                assignments[request_idx] = int(piece_idx)

        return constants.ASSIGN, assignments

    def is_valid_cut(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        existing_points: list[tuple[float, float]],
    ) -> bool:
        """Check if a cut is valid, allowing sequential cuts but preventing close parallel cuts"""
        new_line = LineString([start, end])

        for point in existing_points[:-1]:
            if point not in (start, end):
                if Point(point).distance(new_line) < 0.001:
                    return False

        return True
