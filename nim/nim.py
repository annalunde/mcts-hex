class Nim:
    def __init__(self, num_rocks, max_draw, player_id=1):
        self.player_id = player_id
        self.num_rocks = num_rocks
        self.max_draw = max_draw

    def get_state(self):
        return [self.num_rocks, self.player_id], self.player_id

    def is_legal_action(self, action):
        assert type(action) == int
        draw_num = action
        if draw_num <= 0:
            print(f"Cannot draw less than one rock")
            return False
        if draw_num > self.max_draw:
            print(f"Cannot draw more than {self.max_draw}")
            return False
        if draw_num > self.num_rocks:
            print(f"Cannot draw more than remaining {self.num_rocks}")
            return False
        return True

    def perform_action(self, action):
        if not self.is_legal_action(action):
            raise Exception(f"Illegal move")

        draw_num = action
        self.num_rocks -= draw_num

        self.player_id *= -1


    def get_actions(self):
        if self.is_finished():
            return []
        #print("num rocks", self.num_rocks)
        return range(1, min(self.num_rocks, self.max_draw) + 1)

    def is_finished(self):
        return self.num_rocks == 0
