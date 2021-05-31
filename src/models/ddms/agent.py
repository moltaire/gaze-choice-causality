import numpy as np


class AgentVars(object):
    def __init__(self, **vars):
        self.vars = vars
        for var in vars.keys():
            self.__setattr__(var, vars[var])

    def __repr__(self):
        return str(self.vars)


class Agent(object):
    def __init__(self, agentVars=None):
        self.agentVars = agentVars


