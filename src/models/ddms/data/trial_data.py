class TrialData(object):
    def __init__(self, **data):
        self.data = data
        for datum in data.keys():
            self.__setattr__(datum, data[datum])

    def __repr__(self):
        return f"TrialData: {self.data!r}"
