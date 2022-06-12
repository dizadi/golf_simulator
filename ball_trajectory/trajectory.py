import abc


class Trajectory(abc.ABC):

    @abc.abstractmethod
    def predict(self, input_params):
        pass


class BallTrajectory(Trajectory):
    def __init__(self, ):
        pass

    def predict(self, input_params):
        """Predicts the flight path of a ball with given input parameters"""
        pass

