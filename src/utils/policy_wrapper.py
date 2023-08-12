class PolicyWrapper:
    def get_action(self, state):
        raise NotImplementedError


class Numpy1DArrayPolicy(PolicyWrapper):
    def __init__(self, np_array):
        super(Numpy1DArrayPolicy, self).__init__()
        self.array = np_array

    def get_action(self, state):
        return self.array[state]


if __name__ == "__main__":
    pass
