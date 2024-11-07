import constants
from players.g2.helpers import *

class UnevenCuts:
    def __init__(self, requests, cake_width, cake_len):
        self.cake_width = cake_width
        self.cake_len = cake_len
        self.move_queue = []
        self.direction = 1

        self.total, self.h_sums, self.v_sums = divide_requests(requests)
        self.h_index = 0
        self.v_index = 0

    def move_horizontal(self,pos):
        size = self.cake_len * self.h_sums[self.h_index] / self.total
        self.move_queue.extend(sneak(pos, [pos[0], pos[1] + size], self.cake_width, self.cake_len))
        if pos[0] == 0:
            opposite = self.cake_width
        else:
            opposite = 0
        self.move_queue.append([opposite, round(pos[1] + size, 2)])
        self.h_index += 1

    def move_horizontal_to_vertical(self,pos):
        if pos[0] != 0:  self.direction = -1
        size = self.direction * self.cake_width * self.v_sums[0] / self.total
        new_x = pos[0] + size
        self.move_queue.extend(sneak(pos, [new_x, self.cake_len], self.cake_width, self.cake_len))
        self.move_queue.append([new_x, 0])
        self.v_index = 1

    def move_vertical(self,pos):
        size = self.direction * self.cake_width * self.v_sums[self.v_index] / self.total
        new_x = pos[0] + size
        self.move_queue.extend(sneak(pos, [new_x, pos[1]], self.cake_width, self.cake_len))
        if pos[1] == 0:
            opposite = self.cake_len
        else:
            opposite = 0
        self.move_queue.append([new_x, opposite])
        self.v_index += 1

    def update_queue(self,cur_pos):
        if self.h_index < len(self.h_sums)-1:
            self.move_horizontal(cur_pos)
        elif self.v_index == 0:
            self.move_horizontal_to_vertical(cur_pos)
        elif self.v_index < len(self.v_sums):
            self.move_vertical(cur_pos)
        

    def move(self, turn_number, cur_pos):
        if turn_number == 1:
            return constants.INIT, [0.01, 0]
        
        done = (self.h_index >= len(self.h_sums)) and (self.v_index >= len(self.v_sums))
        
        if turn_number == 2:
            size = self.cake_len * self.h_sums[0] / self.total
            self.move_queue.append([0, size])
            self.move_queue.append([self.cake_width, size])
            self.h_index = 1
        elif len(self.move_queue) == 0 and not done:
            self.update_queue(cur_pos)

        if len(self.move_queue) > 0:
            next_val = self.move_queue.pop(0)
            cut = [round(next_val[0], 2), round(next_val[1], 2)]
            return constants.CUT, cut

        return None