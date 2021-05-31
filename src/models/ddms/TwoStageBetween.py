from ddm import BoundConstant, Drift, Fittable, Model, NoiseConstant

from src.models.ddms.pyddmTwoStage import TwoStagePyDDM


class TwoStageBetween(TwoStagePyDDM):
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
            self.label = "TwoStageBetween"
        else:
            self.label = label

    def compute_stage_two_drift(self, trial):
        """
        Compute stage 2 drift.
        """
        pSum = trial["p0"] + trial["p1"]
        p0n = trial["p0"] / pSum
        p1n = trial["p1"] / pSum

        mSum = trial["m0"] + trial["m1"]
        m0n = trial["m0"] / mSum
        m1n = trial["m1"] / mSum

        dp = p0n - p1n
        dm = m0n - m1n

        wp = self.agentVars.wp

        drift = trial["gp"] * (wp * dp + self.agentVars.eta * (1 - wp) * dm) + trial[
            "gm"
        ] * (self.agentVars.eta * wp * dp + (1 - wp) * dm)

        last_stage_bias = (
            trial["presentation01"]
            * self.agentVars.b_last
            * ((trial["last_stage_favours"] * (-2)) + 1)  # recodes 0, 1 to 1, -1
        )

        if trial["presentation"] == "alternatives":
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
            drift=TwoStageBetweenDrift(
                eta=Fittable(minval=0, maxval=1),
                wp=Fittable(minval=0, maxval=1),
                v=Fittable(minval=0, maxval=30),
                b_last=Fittable(minval=-1, maxval=1),
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
            "gp",
            "gm",
            "presentation01",
            "last_stage_favours",
        ]
        return model


class TwoStageBetweenDrift(Drift):
    """Drift class for the Two-Stage Within-alternative integration pyDDM."""

    name = "Gaze-biased between-alternative integration Stage II drift"
    required_parameters = ["wp", "eta", "v", "b_last"]
    required_conditions = [
        "p0",
        "p1",
        "m0",
        "m1",
        "gp",
        "gm",
        "last_stage_favours",
        "presentation01",
    ]

    def get_drift(self, conditions, **kwargs):

        pSum = conditions["p0"] + conditions["p1"]
        p0n = conditions["p0"] / pSum
        p1n = conditions["p1"] / pSum
        mSum = conditions["m0"] + conditions["m1"]
        m0n = conditions["m0"] / mSum
        m1n = conditions["m1"] / mSum

        dp = p0n - p1n
        dm = m0n - m1n

        drift = conditions["gp"] * (
            self.wp * dp + self.eta * (1 - self.wp) * dm
        ) + conditions["gm"] * (self.eta * self.wp * dp + (1 - self.wp) * dm)

        last_stage_bias = (
            conditions["presentation01"]
            * self.b_last
            * ((conditions["last_stage_favours"] * (-2)) + 1)  # recodes 0, 1 to 1, -1
        )

        return self.v * (drift + last_stage_bias)
