import numpy as np

class FitnessMethodInterface:

    def __init__(self, scale_fitness = True):
        self.scale_fitness = scale_fitness

    def compute(self, x_e, y_e, x_m, y_m):
        print("using undefined function")

class FitnessMethodRMSE(FitnessMethodInterface):

    def __init__(self, n_points = None, x_def_range = None, scale_fitness = True):
        self.n = n_points
        self.x_def = None
        self.x_def_range = x_def_range
        self.scale_fitness=scale_fitness
        FitnessMethodInterface.__init__(self, scale_fitness=scale_fitness)

    def compute(self, x_e, y_e, x_m, y_m):
        # compute x array on which the data sets shall be mapped to in order to compute the RMSE on the
        # same definition range
        if self.x_def is None:
            if self.x_def_range is None:
                x_min = np.max([np.min(x_e), np.min(x_m)])
                x_max = np.min([np.max(x_e), np.max(x_m)])
                self.x_def_range = [x_min, x_max]
            self.x_def = np.linspace(self.x_def_range[0], self.x_def_range[1], self.n, endpoint=True)

        y_e_mapped = np.interp(self.x_def, x_e, y_e)
        y_m_mapped = np.interp(self.x_def, x_m, y_m)

        rmse = np.sqrt(((y_e_mapped - y_m_mapped) ** 2).mean())
        if self.scale_fitness == 'mean' or self.scale_fitness == True:
            return rmse/np.abs(np.mean(y_e_mapped))
        elif self.scale_fitness == 'minmax':
            return rmse/np.abs(y_e_mapped[-1]-y_e_mapped[0])
        elif self.scale_fitness == 'interquartile':
            return rmse
        else:
            return rmse


class FitnessMethodThreshold(FitnessMethodInterface):

    def __init__(self, threshold_type, threshold_value = None, threshold_range = None, scale_fitness = True):

        FitnessMethodInterface(scale_fitness)

        threshold_types = ['upper', 'lower', 'range_minmax']
        if threshold_type not in threshold_types:
            print("wrong threshold type, available types are:", threshold_types)
        self.type = threshold_type
        self.value = threshold_value
        self.range = threshold_range
        self.scale_fitness=scale_fitness

    def compute(self, x_e, y_e, x_m, y_m):

        x_e_threshold = None
        x_m_threshold = None
        if self.type == "upper" or self.type == "lower":
            x_e_threshold = self.simple_threshold(self.type, self.value, x_e, y_e)
            x_m_threshold = self.simple_threshold(self.type, self.value, x_m, y_m)

        if self.type == "range_minmax":
            x_e_threshold_lower = self.simple_threshold("lower", self.range[0], x_e, y_e)
            x_e_threshold_upper = self.simple_threshold("upper", self.range[1], x_e, y_e)
            x_m_threshold_lower = self.simple_threshold("lower", self.range[0], x_m, y_m)
            x_m_threshold_upper = self.simple_threshold("upper", self.range[1], x_m, y_m)

            # result is the smallest value in x when the range was left
            if x_e_threshold_lower is not None and x_e_threshold_upper is not None:
                x_e_threshold = np.min(x_e_threshold_lower, x_e_threshold_upper)
            if x_m_threshold_lower is not None and x_m_threshold_upper is not None:
                x_m_threshold = np.min(x_m_threshold_lower, x_m_threshold_upper)

        # check if the experimental data returns a valid threshold evaluation
        if len(x_e_threshold) == 0:
            print("rethink your fitness method choice")

        # if the model data never reaches the threshold, return maximal deviation w.r.t. the
        # experimental value, i.e. maximal model x-value minus experimental threshold position
        if len(x_m_threshold) == 0:
            x_m_max_distance = np.abs(np.max(x_m) - x_e_threshold[0])
            if self.scale_fitness == True:
                return np.abs(x_m_max_distance / x_e_threshold[0])
            else:
                return x_m_max_distance
        if self.scale_fitness == True:
            return np.abs((x_e_threshold[0] - x_m_threshold[0])/x_e_threshold[0])
        return np.abs(x_e_threshold[0] - x_m_threshold[0])


    def simple_threshold(self, t, v, x, y):

        indices = None
        if t == "upper":
            indices = np.where(y > v)
        if t == "lower":
            indices = np.where(y < v)

        if len(indices) > 0:
            result_index = indices[0]
            result_x = x[result_index]
        else:
            print("threshold was not reached")
            result_x = None
        return result_x

