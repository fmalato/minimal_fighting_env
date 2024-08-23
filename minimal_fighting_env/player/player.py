class Player:
    def __init__(self, color: tuple, max_hp: int = 3):
        self.position = {"x": None, "y": None}
        self.color = color
        self.stunned_frames = 0
        self.max_hp = max_hp
        self.hp = max_hp
        self.damaged_frames = 0

    def get_state(self):
        return [self.position["x"], self.position["y"], self.hp, self.stunned_frames, self.damaged_frames]

    def get_dict_state(self):
        return {
            "x": self.position["x"],
            "y": self.position["y"],
            "hp": self.hp,
            "stunned": self.stunned_frames,
            "damaged": self.damaged_frames
        }

    def get_position(self):
        return self.position

    def set_position(self, x: int, y: int):
        self.position = {"x": x, "y": y}

    def get_color(self):
        return self.color

    def get_hp(self):
        return self.hp

    def set_hp(self, value):
        self.hp = value

    def get_stunned(self):
        return self.stunned_frames

    def set_stunned(self, value):
        self.stunned_frames = value

    def decrease_stunned(self):
        if self.stunned_frames > 0:
            self.stunned_frames -= 1

    def get_damaged(self):
        return self.damaged_frames

    def set_damaged(self, value):
        self.damaged_frames = value

    def decrease_damaged(self):
        if self.damaged_frames > 0:
            self.damaged_frames -= 1

    def reset(self, x: int, y: int):
        self.position = {"x": x, "y": y}
        self.stunned_frames = 0
        self.damaged_frames = 0
        self.hp = self.max_hp
