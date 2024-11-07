import constants
import numpy as np
from players.g2.helpers import *

class EvenCuts:
    def __init__(self, requests, cake_width, cake_len):
        self.phase = "HORIZONTAL"
        self.direction = ""
        self.cake_width = cake_width
        self.cake_len = cake_len

        best = (max(requests) + min(requests)) / 2
        
        n = len(requests)
        w = int(np.sqrt(n))
        self.s_x = self.cake_width / w
        self.s_y = best / self.s_x

        self.move_queue = []

    def even_cuts(self, pos):
            """
            Adds moves to the merge queue that will cut the cake into even slices.
            """

            if self.phase == "HORIZONTAL" and pos[1] + self.s_y >= self.cake_len:
                self.phase = "VERTICAL"
                if pos[0] == 0:
                    new_x = self.s_x
                else:
                    new_x = self.cake_width - self.s_x
                    self.direction = "RIGHT"
                self.move_queue.extend(sneak(pos, [new_x, self.cake_len], self.cake_width, self.cake_len))
                self.move_queue.append([new_x, 0])

                return

            if self.phase == "HORIZONTAL":
                self.move_queue.extend(sneak(pos, [pos[0], pos[1] + self.s_y], self.cake_width, self.cake_len))
                if pos[0] == 0:
                    opposite = self.cake_width
                else:
                    opposite = 0
                self.move_queue.append([opposite, round(pos[1] + self.s_y, 2)])

            else:
                if self.direction == "RIGHT":
                    new_x = pos[0] - self.s_x
                else:
                    new_x = pos[0] + self.s_x

                if new_x <= 0 or new_x >= self.cake_width:
                    self.phase = "DONE"
                    return

                self.move_queue.extend(sneak(pos, [new_x, pos[1]], self.cake_width, self.cake_len))
                if pos[1] == 0:
                    opposite = self.cake_len
                else:
                    opposite = 0
                self.move_queue.append([new_x, opposite])

            return

    def move(self, turn_number, cur_pos):
            if turn_number == 1:
                return constants.INIT, [0.01, 0]
            
            if turn_number == 2:
                self.move_queue.append([0, self.s_y])
                self.move_queue.append([self.cake_width, self.s_y])
            elif len(self.move_queue) == 0 and self.phase != "DONE":
                self.even_cuts(cur_pos)

            if len(self.move_queue) > 0:
                next_val = self.move_queue.pop(0)
                cut = [round(next_val[0], 2), round(next_val[1], 2)]
                return constants.CUT, cut

            return None