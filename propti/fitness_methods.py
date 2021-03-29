import sys
import logging
import numpy as np


class FitnessMethodInterface:

    def __init__(self, scale_fitness=True):
        self.scale_fitness = scale_fitness

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        print("using undefined function")


class FitnessMethodRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        # compute x array on which the data sets shall be mapped to,
        # in order to compute the RMSE on the same definition range
        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        rmse = np.sqrt(((y_e_mapped - y_m_mapped) ** 2).mean())
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(y_e_mapped[-1] - y_e_mapped[0])
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodRangeRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None,
                 y_relative_range=None, scale_fitness=True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        if y_relative_range is None:
            self.y_relative_range = 0.05
        else:
            self.y_relative_range=abs(y_relative_range)
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        # compute x array on which the data sets shall be mapped to, in order
        #  to compute the RMSE on the same definition range
        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        y_rmse = np.zeros(y_e_mapped.shape)
        for i, value in enumerate(y_e_mapped):
            if (y_e_mapped[i]*(1-self.y_relative_range)) <= y_m_mapped[i] <= (y_e_mapped[i]*(1+self.y_relative_range)):
                y_rmse[i] = 0
            else:
                y_rmse[i] = (y_e_mapped[i] - y_m_mapped[i]) ** 2
        rmse = np.sqrt(y_rmse.mean())
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(y_e_mapped[-1] - y_e_mapped[0])
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodBandRMSE(FitnessMethodInterface):

    def __init__(self, n_points=None, x_def_range=None, scale_fitness=True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness = scale_fitness
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, y2_e, x_m, y_m):
        # compute x array on which the data sets shall be mapped to,
        # in order to compute the RMSE on the same definition range.
        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0],
                                     self.x_def_range[1],
                                     self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_e_mapped_b2 = np.interp(self.x_def, x_e, y2_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)
        y_rmse = np.zeros(y_e_mapped.shape)
        for i, value in enumerate(y_e_mapped):
            if np.min((y_e_mapped[i], y_e_mapped_b2[i])) <= y_m_mapped[i] <= \
                    np.max((y_e_mapped[i], y_e_mapped_b2[i])):
                y_rmse[i] = 0
            else:
                y_rmse[i] = np.min((((y_e_mapped[i] - y_m_mapped[i]) ** 2),
                                   ((y_e_mapped_b2[i] - y_m_mapped[i]) ** 2)))
        rmse = np.sqrt(y_rmse.mean())
        if self.scale_fitness == 'mean' or self.scale_fitness is True:
            return rmse / np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse / np.abs(y_e_mapped[-1] - y_e_mapped[0])
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodThreshold(FitnessMethodInterface):

    def __init__(self, threshold_type, threshold_target_value=None,
                 threshold_value=None, threshold_range=None,
                 scale_fitness=True):

        super().__init__(scale_fitness)

        threshold_types = ['upper', 'lower', 'range_minmax']
        if threshold_type not in threshold_types:
            print("wrong threshold type, available types are:", threshold_types)
            # TODO handle this?
        self.type = threshold_type
        self.threshold_target_value = threshold_target_value
        self.value = threshold_value
        self.range = threshold_range
        self.scale_fitness = scale_fitness

    def compute(self, x_e, y_e, y2_e, x_m, y_m):

        x_e_threshold = None
        x_m_threshold = None
        if self.type == "upper" or self.type == "lower":
            # only needed for experimental data if no target value was specified
            if self.threshold_target_value is None:
                x_e_threshold = self.simple_threshold(self.type,
                                                      self.value,
                                                      x_e,
                                                      y_e)
            else:
                x_e_threshold = self.threshold_target_value
            x_m_threshold = self.simple_threshold(self.type,
                                                  self.value,
                                                  x_m,
                                                  y_m)

        if self.type == "range_minmax":
            # only needed for experimental data if no target value was specified
            if self.threshold_target_value is None:
                x_e_threshold_lower = self.simple_threshold("lower",
                                                            self.range[0],
                                                            x_e, y_e)
                x_e_threshold_upper = self.simple_threshold("upper",
                                                            self.range[1],
                                                            x_e, y_e)
            x_m_threshold_lower = self.simple_threshold("lower",
                                                        self.range[0],
                                                        x_m, y_m)
            x_m_threshold_upper = self.simple_threshold("upper",
                                                        self.range[1],
                                                        x_m, y_m)

            # check if target value was explicitly passed
            if self.threshold_target_value is not None:
                x_e_threshold = self.threshold_target_value
            else:
                # result is the smallest value in x when the range was left
                if x_e_threshold_lower is not None and \
                        x_e_threshold_upper is not None:
                    x_e_threshold = np.min(x_e_threshold_lower,
                                           x_e_threshold_upper)
            if x_m_threshold_lower is not None and \
                    x_m_threshold_upper is not None:
                x_m_threshold = np.min(x_m_threshold_lower, x_m_threshold_upper)

        # check if the experimental data returns a valid threshold evaluation
        if x_e_threshold is None:
            print("ERROR: rethink your fitness method choice")
            logging.error("rethink your fitness method choice")
            sys.exit(1)

        # if the model data never reaches the threshold,
        # return maximal deviation w.r.t. the experimental value,
        # i.e. maximal model x-value minus experimental threshold position
        if x_m_threshold is None:
            x_m_max_distance = np.abs(np.max(x_m) - x_e_threshold)
            if self.scale_fitness:
                return np.abs(x_m_max_distance / x_e_threshold)
            else:
                return x_m_max_distance

        if self.scale_fitness:
            return np.abs((x_e_threshold - x_m_threshold) / x_e_threshold)

        return np.abs(x_e_threshold - x_m_threshold)

    def simple_threshold(self, t, v, x, y):

        indices = None
        if t == "upper":
            indices = np.where(y > v)
        if t == "lower":
            indices = np.where(y < v)

        if len(indices[0]) > 0:
            result_index = indices[0][0]
            result_x = x[result_index]
        else:
            print("threshold was not reached")
            result_x = None

        return result_x
