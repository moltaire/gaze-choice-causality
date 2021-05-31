import numpy as np


class Fixation(object):
    def __init__(self, duration, target):
        self.duration = duration
        self.target = target

    def __len__(self):
        return self.duration

    def __repr__(self):
        return f"Fixation to {self.target!r} with duration {self.duration!r}"


class FixationSequence(object):
    def __init__(self, fixations):
        self.fixations = fixations

    def __iter__(self):
        for fixation in self.fixations:
            yield fixation

    def __len__(self):
        return len(self.fixations)

    def __repr__(self):
        return "Fixation sequence:\n" + "".join(
            ["\t{}\n".format(fixation) for fixation in self.fixations]
        )

    @property
    def duration(self):
        return np.sum([fixation.duration for fixation in self.fixations])


class FixationTarget(object):
    def __init__(self, **dimensions):
        self.dimensions = dimensions
        for dimension in dimensions.keys():
            self.__setattr__(dimension, dimensions[dimension])

    def __repr__(self):
        return f"FixationTarget: {self.dimensions!r}"

