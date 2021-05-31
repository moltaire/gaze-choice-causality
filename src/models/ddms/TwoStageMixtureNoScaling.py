from ddm import BoundConstant, Drift, Fittable, Model, NoiseConstant

from src.models.ddms.pyddmTwoStage import TwoStagePyDDM


class TwoStageMixtureNoScaling(TwoStagePyDDM):
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
            self.label = "TwoStageMixtureNoScaling"
        else:
            self.label = label

    def compute_stage_two_drift(self, trial):
        """
        Compute stage 2 drift.
        """

        # Between alternative component
        pSum = trial["p0"] + trial["p1"]
        p0n = trial["p0"] / pSum
        p1n = trial["p1"] / pSum

        mSum = trial["m0"] + trial["m1"]
        m0n = trial["m0"] / mSum
        m1n = trial["m1"] / mSum

        dp = p0n - p1n
        dm = m0n - m1n

        wp = self.agentVars.wp

        # Set eta to 1 in trials with alternative-wise presentation
        # eta_trial = (1 - trial["presentation01"]) + trial[
        #     "presentation01"
        # ] * self.agentVars.eta
        eta_trial = self.agentVars.eta

        drift_between = trial["gp"] * (wp * dp + eta_trial * (1 - wp) * dm) + trial[
            "gm"
        ] * (eta_trial * wp * dp + (1 - wp) * dm)

        # Within alternative component
        eu0 = trial["p0"] * trial["m0"] ** self.agentVars.alpha
        eu1 = trial["p1"] * trial["m1"] ** self.agentVars.alpha

        # Set theta to 1 in attribute-wise presentation trials
        # theta_trial = (1 - trial["presentation01"]) * self.agentVars.theta + (
        #     trial["presentation01"]
        # )
        theta_trial = self.agentVars.theta

        drift_within = trial["g0"] * (eu0 - theta_trial * eu1) + trial["g1"] * (
            theta_trial * eu0 - eu1
        )

        last_stage_bias = last_stage_bias = self.agentVars.b_last * (
            (trial["last_stage_favours"] * (-2)) + 1
        )  # recodes 0, 1 to 1, -1

        # Compute mixture weight based on trial type
        w_between = (
            trial["presentation01"] * self.agentVars.w_between_attwise
            + (1 - trial["presentation01"]) * self.agentVars.w_between_altwise
        )

        drift = (
            (1 - w_between) * drift_within + w_between * drift_between + last_stage_bias
        )
        return drift

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
            drift=TwoStageMixtureDrift(
                v=Fittable(minval=0, maxval=30),
                alpha=Fittable(minval=0, maxval=5),
                theta=Fittable(minval=0, maxval=1),
                wp=Fittable(minval=0, maxval=1),
                eta=Fittable(minval=0, maxval=1),
                w_between_altwise=Fittable(minval=0, maxval=1),
                w_between_attwise=Fittable(minval=0, maxval=1),
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
            "g0",
            "g1",
            "presentation01",
            "last_stage_favours",
        ]
        return model


class TwoStageMixtureDrift(Drift):
    """Drift class for the Two-Stage within- and between-alternative integration mixture pyDDM."""

    name = "Gaze-biased mixture of between- and within-alternative integration Stage II drifts"
    required_parameters = [
        "v",
        "alpha",
        "theta",
        "wp",
        "eta",
        "w_between_altwise",
        "w_between_attwise",
        "b_last",
    ]
    required_conditions = [
        "p0",
        "p1",
        "m0",
        "m1",
        "gp",
        "gm",
        "g0",
        "g1",
        "last_stage_favours",
        "presentation01",
    ]

    def get_drift(self, conditions, **kwargs):

        # Between alternatives component
        pSum = conditions["p0"] + conditions["p1"]
        p0n = conditions["p0"] / pSum
        p1n = conditions["p1"] / pSum
        mSum = conditions["m0"] + conditions["m1"]
        m0n = conditions["m0"] / mSum
        m1n = conditions["m1"] / mSum

        # Between alternative component
        pSum = conditions["p0"] + conditions["p1"]
        p0n = conditions["p0"] / pSum
        p1n = conditions["p1"] / pSum

        mSum = conditions["m0"] + conditions["m1"]
        m0n = conditions["m0"] / mSum
        m1n = conditions["m1"] / mSum

        dp = p0n - p1n
        dm = m0n - m1n

        # Set eta to 1 in trials with alternative-wise presentation
        # eta_trial = (1 - conditions["presentation01"]) + conditions[
        #     "presentation01"
        # ] * self.eta
        eta_trial = self.eta

        drift_between = conditions["gp"] * (
            self.wp * dp + eta_trial * (1 - self.wp) * dm
        ) + conditions["gm"] * (eta_trial * self.wp * dp + (1 - self.wp) * dm)

        # Within alternative component
        eu0 = conditions["p0"] * conditions["m0"] ** self.alpha
        eu1 = conditions["p1"] * conditions["m1"] ** self.alpha

        # Set theta to 1 in attribute-wise presentation trials
        # theta_trial = (1 - conditions["presentation01"]) * self.theta + (
        #     conditions["presentation01"]
        # )
        theta_trial = self.theta

        drift_within = conditions["g0"] * (eu0 - theta_trial * eu1) + conditions[
            "g1"
        ] * (theta_trial * eu0 - eu1)

        last_stage_bias = last_stage_bias = self.b_last * (
            (conditions["last_stage_favours"] * (-2)) + 1
        )  # recodes 0, 1 to 1, -1

        # Compute mixture weight based on trial type
        w_between = (
            conditions["presentation01"] * self.w_between_attwise
            + (1 - conditions["presentation01"]) * self.w_between_altwise
        )

        drift = self.v * (
            (1 - w_between) * drift_within + w_between * drift_between + last_stage_bias
        )

        return drift