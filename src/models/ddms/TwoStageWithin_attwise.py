import numpy as np
from ddm import BoundConstant, Drift, Fittable, Model, NoiseConstant

from src.models.ddms.pyddmTwoStage import TwoStagePyDDM


class TwoStageWithin_attwise(TwoStagePyDDM):
    def __init__(self, dt=0.01, dx=0.01, T_dur=3, agentVars=None, label=None):
        """[summary]

        Parameters
        ----------
        Agent : [type]
            [description]
        agentVars : [type], optional
            [description], by default None
        """
        super().__init__(agentVars=agentVars)
        self.dt = dt
        self.dx = dx
        self.T_dur = T_dur
        if label is None:
            self.label = "TwoStageWithin"
        else:
            self.label = label

    def compute_stage_two_drift(self, trial):
        """
        Compute stage 2 drift.
        """
        eu0 = trial["p0"] * trial["m0"] ** self.agentVars.alpha
        eu1 = trial["p1"] * trial["m1"] ** self.agentVars.alpha

        drift = trial["g0"] * (eu0 - self.agentVars.theta * eu1) + trial["g1"] * (
            self.agentVars.theta * eu0 - eu1
        )

        last_stage_bias = (
            (1 - trial["presentation01"])
            * self.agentVars.b_last
            * ((trial["last_stage_favours"] * (-2)) + 1)  # recodes 0, 1 to 1, -1
        )
        if trial["presentation"] == "attributes":
            assert last_stage_bias == 0

        return drift + last_stage_bias

    def build_model(self):
        """Builds the pyDDM model for this agent.

        Args:
            dx (float, optional): Resolution of evidence space. Defaults to 0.01.
            dt (float, optional): Resolution of time domain. Defaults to 0.01.
            T_dur (int, optional): Maximum trial duration. Defaults to 5.

        Returns:
            pyDDM model instance
        """
        model = Model(
            name=self.label,
            drift=TwoStageWithinDrift(
                theta=1,
                alpha=Fittable(minval=0, maxval=5),
                v=Fittable(minval=0, maxval=30),
                b_last=0,
            ),
            noise=NoiseConstant(noise=Fittable(minval=0.001, maxval=4)),
            bound=BoundConstant(B=1),
            dx=self.dx,
            dt=self.dt,
            T_dur=self.T_dur,
        )
        self.model = model
        self.required_conditions = [
            "p0",
            "p1",
            "m0",
            "m1",
            "g0",
            "g1",
            "presentation01",
            "last_stage_favours",
        ]
        return model


class TwoStageWithinDrift(Drift):
    """Drift class for the Two-Stage Within-alternative integration pyDDM."""

    name = "Gaze-biased within-alternative integration Stage II drift"
    required_parameters = ["alpha", "theta", "v", "b_last"]
    required_conditions = [
        "p0",
        "p1",
        "m0",
        "m1",
        "g0",
        "g1",
        "presentation01",
        "last_stage_favours",
    ]

    def get_drift(self, conditions, **kwargs):
        eu0 = conditions["p0"] * conditions["m0"] ** self.alpha
        eu1 = conditions["p1"] * conditions["m1"] ** self.alpha

        drift = conditions["g0"] * (eu0 - self.theta * eu1) + conditions["g1"] * (
            self.theta * eu0 - eu1
        )

        last_stage_bias = (
            (1 - conditions["presentation01"])
            * self.b_last
            * ((conditions["last_stage_favours"] * (-2)) + 1)  # recodes 0, 1 to 1, -1
        )

        return self.v * (drift + last_stage_bias)
