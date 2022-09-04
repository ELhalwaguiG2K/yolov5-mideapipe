from json import JSONEncoder


class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class Person:
    def __init__(self, id: int, landmarks: [Landmark], width: int, height: int):
        self.id = id
        self.landmarks = landmarks
        self.resolution_X = width
        self.resolution_Y = height


class Root:
    def __init__(self, persons: [Person]):
        self.persons = persons


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
