import random
import math
from browser import document, window  # type: ignore

# import chartjs module
chartjs = window.Chart
# import mathjs module
mathjs = window.math


# # --------------------------------------------- gaussianprocessregression.py ----------------------------------------------------------------

# from browser import window  # type: ignore
# import math

# # import mathjs module
# mathjs = window.math


class GaussianProcessRegressionPy:
    def __init__(self, length_scale=1, prior_variance=1, data_noise=0):
        self._length_scale = length_scale

        self._sigma_prior_squared = prior_variance
        self._sigma_noise_squared = data_noise

        self._data_X = None
        self._data_Y = None
        self._prior_mean = None
        self._prior_mean_function = None

        self.prediction = []

    @property
    def _sigma_squared(self):    
        return self._sigma_prior_squared - self._sigma_noise_squared

    def set_prior_mean(self, func, axis):
        self._prior_mean = [func(x) for x in axis]
        self._prior_mean_function = func

    @property
    def prior(self):
        return [self._prior_mean, [self._sigma_prior_squared for _ in range(len(self._prior_mean))]]

    @property
    def data_Y_prior(self):
        return [self.data_Y[i] - self._prior_mean_function(self.data_X[i]) for i in range(len(self.data_X))]

    def _k_RBF(self, xi, xj):
        noise = (self._sigma_noise_squared if xi == xj else 0)
        return self._sigma_squared * math.exp(-(xi-xj)**2/(2 * self._length_scale**2)) + noise

    def _calc_Sigma(self, X_t):
        X_new = self.data_X + X_t
        return [[self._k_RBF(xi,xj) for xj in X_new] for xi in X_new]

    def _calc_gp(self, covariance_matrix):  # covariance_matrix is a n+m x n+m matrix
        # data_X and data_Y are n vectors
        if self.data_X:
            n = len(self.data_X)

            K = [row[:n] for row in covariance_matrix[:n]]  # nxn matrix
            Kstar = [row[n:] for row in covariance_matrix[:n]]  # nxm matrix
            KstarT = [row[:n] for row in covariance_matrix[n:]]  # mxn matrix
            Kstarstar = [row[n:] for row in covariance_matrix[n:]]  # mxm matrix

            Kinv = mathjs.inv(K)  # nxn matrix --> need javascript mathjs library for inversion

            KstarTKinvY = mathjs.multiply(mathjs.multiply(KstarT,Kinv),self.data_Y_prior)
            KstarTKinvKstar = mathjs.multiply(mathjs.multiply(KstarT,Kinv),Kstar)

            mu_t = [KstarTKinvY[i] + self._prior_mean[i] for i in range(len(self._prior_mean))]
            variance_t = [Kstarstar[i][i] - KstarTKinvKstar[i][i] for i in range(len(Kstarstar))]
        else:
            mu_t = list(self._prior_mean)
            variance_t = [covariance_matrix[i][i] for i in range(len(covariance_matrix))]

        return [mu_t, variance_t]


    def set_data(self, X, Y):
        self.data_X = X
        self.data_Y = Y

    def predict(self, x_axis):
        if self._prior_mean is None:
            self.set_prior_mean(lambda _: 0, x_axis)
        self.prediction = self._calc_gp(self._calc_Sigma(x_axis))



# # --------------------------------------------- brythonchart.py ----------------------------------------------------------------

# from browser import document, window  # type: ignore

# # import chartjs module
# chartjs = window.Chart


# chartjs 
class BrythonChartJS:
    def __init__(self, div_id):

        self._datapoints = []

        self._prediction = []
        self._prediction_add_variance = []
        self._prediction_sub_variance = []

        self._prior = []
        self._prior_add_variance = []
        self._prior_sub_variance = []

        self._chart = chartjs.new(document[div_id].getContext('2d'),{
            "data": {
                "datasets": [{
                    "type": 'bubble',
                    "label": 'Dataset',
                    "data": self._datapoints,
                    "backgroundColor": 'rgb(99, 255, 132)'
                },{
                    "type": 'line',
                    "label": 'GP Prediction',
                    "data": self._prediction,
                    "backgroundColor": 'rgba(132, 99, 255, 1)'
                },{
                    "type": 'line',
                    "label": 'GP Variance',
                    "fill": '+1',
                    "data": self._prediction_add_variance,
                    "backgroundColor": 'rgba(132, 99, 255, 0.2)',
                    "pointRadius": 0,
                    "pointHitRadius": 5

                },{
                    "type": 'line',
                    "label": 'GP Variance',
                    "fill": "-1",
                    "data": self._prediction_sub_variance,
                    "backgroundColor": 'rgba(132, 99, 255, 0.2)',
                    "pointRadius": 0,
                    "pointHitRadius": 5
                },{
                    "type": 'line',
                    "label": 'Prior',
                    "data": self._prior,
                    "backgroundColor": 'rgba(32, 99, 255, 0.3)'
                },{
                    "type": 'line',
                    "label": 'Prior Variance',
                    "fill": '+1',
                    "data": self._prior_add_variance,
                    "backgroundColor": 'rgba(32, 99, 255, 0.1)',
                    "pointRadius": 0,
                    "pointHitRadius": 5
                },{
                    "type": 'line',
                    "label": 'Prior Variance',
                    "fill": "-1",
                    "data": self._prior_sub_variance,
                    "backgroundColor": 'rgba(32, 99, 255, 0.1)',
                    "pointRadius": 0,
                    "pointHitRadius": 5
                }]
            },
            "options": {
                "plugins": {
                    "legend": {
                        "labels": {"filter": lambda item, chart: self.legendFilter(item, chart)},
                        "onClick": lambda  e, item, *args: self.click_legend(e, item, *args)
                    },
                },
                "scales": {
                    "y": {
                        "suggestedMax": 50,
                        "suggestedMin": 0,
                    },
                    "x": {
                        "suggestedMin": 0,
                        "suggestedMax": 10
                    }
                },
                "responsive": True,
                "onClick": lambda e, *args: self.click_chart(e, *args)
            },

        })

        self.pre_update_hook = None

    @property
    def datapoints(self):
        return self._datapoints

    @datapoints.setter
    def datapoints(self, val):
        """setter method automatically transforms data into chart version
        
        input: val = [[x1, x2, x3, ..., xn], [y1, y2, y3, ..., yn]]
        output: self._datapoints = [{"x": x1, "y": y1, "r": 10}, ..., {"x": xn, "y": yn, "r": 10}]
        """
        inputT = [[val[j][i] for j in range(len(val))] for i in range(len(val[0]))]
        self._datapoints = [{"x": el[0], "y": el[1], "r": 10} for el in inputT]

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, val):
        """setter method automatically transforms data into chart version; adds variances
        
        input: val = [[x1, x2, ..., xk], [mu1, mu2, ..., muk], [var1, var2, ..., vark]]]
        outputs: 
            self._prediction = [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ..., {"x": xn, "y": yn}]
            self._prediction_add_variance = [{"x": x1, "y": y1 + var}, {"x": x2, "y": y2 + var}, ..., {"x": xn, "y": yn + var}]
            self._prediction_sub_variance = [{"x": x1, "y": y1 - var}, {"x": x2, "y": y2 - var}, ..., {"x": xn, "y": yn - var}]
        """

        x_mu_var = list(zip(*val))

        self._prediction = [{"x": el[0], "y": el[1]} for el in x_mu_var]
        self._prediction_add_variance = [{"x": el[0], "y": el[1] + el[2]} for el in x_mu_var]
        self._prediction_sub_variance = [{"x": el[0], "y": el[1] - el[2]} for el in x_mu_var]

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, val):
        """setter method automatically transforms data into chart version; adds variances
        
        input: val = [[x1, x2, ..., xk], [mu1, mu2, ..., muk], [var1, var2, ..., vark]]]
        outputs: 
            self._prior = [{"x": x1, "y": y1}, {"x": x2, "y": y2}, ..., {"x": xn, "y": yn}]
            self._prior_add_variance = [{"x": x1, "y": y1 + var}, {"x": x2, "y": y2 + var}, ..., {"x": xn, "y": yn + var}]
            self._prior_sub_variance = [{"x": x1, "y": y1 - var}, {"x": x2, "y": y2 - var}, ..., {"x": xn, "y": yn - var}]
        """
        x_mu_var = list(zip(*val))

        self._prior = [{"x": el[0], "y": el[1]} for el in x_mu_var]
        self._prior_add_variance = [{"x": el[0], "y": el[1] + el[2]} for el in x_mu_var]
        self._prior_sub_variance = [{"x": el[0], "y": el[1] - el[2]} for el in x_mu_var]

    # used to train the gp when points have been removed or added interactively
    @property
    def datapoints_inputform(self):
        return [[el["x"] for el in self._datapoints],[el["y"] for el in self._datapoints]]

    def add_datapoint(self, data):  # takes data as [x, y]
        self._datapoints += [{"x": data[0], "y": data[1], "r": 10}]

    def update_chart(self):
        datasets = self._chart["data"]["datasets"]
        for ds in datasets:
            if datasets.index(ds) == 0:
                ds["data"] = self._datapoints
                self._chart.update("none")
            elif datasets.index(ds) == 1:
                ds["data"] = self._prediction
            elif datasets.index(ds) == 2:
                ds["data"] = self._prediction_add_variance
            elif datasets.index(ds) == 3:
                ds["data"] = self._prediction_sub_variance
            elif datasets.index(ds) == 4:
                ds["data"] = self._prior
            elif datasets.index(ds) == 5:
                ds["data"] = self._prior_add_variance
            elif datasets.index(ds) == 6:
                ds["data"] = self._prior_sub_variance

            self._chart.update()


    # args catches whatever else is handed over
    def click_chart(self, event, *args):

        # if clicked at chart element, remove if its a datapoint
        element = self._chart.getElementsAtEventForMode(event, 'point', self._chart["options"])
        if len(element) > 0:
            for el in element:
                if el.datasetIndex == 0:
                    self._datapoints.pop(el.index)

        # else prepare to add a new datapoint
        else:
            canvasPosition = chartjs.helpers.getRelativePosition(event, self._chart)
            new_x = self._chart.scales.x.getValueForPixel(canvasPosition.x)
            new_y = self._chart.scales.y.getValueForPixel(canvasPosition.y)

            # check if datapoint with that x exists, if yes, change its y coordinate, if not, add datapoint
            existing_x = [el["x"] for el in self._datapoints]
            if new_x in existing_x:
                self._datapoints[existing_x.index(new_x)]["y"] = new_y
            else:
                self.add_datapoint([new_x, new_y])

        # do any other custom calculations (recalculate gp)
        if self.pre_update_hook is not None:
            self.pre_update_hook()

        # and update the chart
        self.update_chart()


    def click_legend(self, event, legendItem, legend):
        index = legendItem.datasetIndex
        # Prediction Variance
        if (index == 2 or index == 3):
            ci = legend.chart
            for meta in [ci.getDatasetMeta(2), ci.getDatasetMeta(3)]:
                if meta.hidden:
                    ci.show(index)
                    meta.hidden = False
                else:
                    ci.hide(index)
                    meta.hidden = True

            ci.update()
        # Prior
        elif (index == 4 or index == 5 or index == 6):
            ci = legend.chart
            for meta in [ci.getDatasetMeta(4), ci.getDatasetMeta(5), ci.getDatasetMeta(6)]:
                if meta.hidden:
                    meta.hidden = False
                else:
                    meta.hidden = True

            ci.update()
        else:
            # Do the original logic
            chartjs.defaults.plugins.legend.onClick(event, legendItem, legend)


    def legendFilter(self, item, chart):
        return False if ((item.datasetIndex == 3) or (item.datasetIndex == 5) or (item.datasetIndex == 6)) else True


# # --------------------------------------------- brython_gp.py ----------------------------------------------------------------

# import random
# from brython_GP.gaussianprocessregression import GaussianProcessRegressionPy
# from brython_GP.brythonchart import BrythonChartJS


class RandomDataPointsPy:
    def __init__(self, nr_pts=10):
        self.X = [0.2]
        self.Y = [random.gauss(10,2)]

        for i in range(1,nr_pts):
            self.X += [random.gauss(i,0.2)]
            self.Y += [self.Y[i-1] + random.gauss(2,5) + 2]

    @property
    def XY(self):
        return [{"x": xy[0], "y": xy[1], "r": 10} for xy in zip(self.X, self.Y)]


datapoints = RandomDataPointsPy()

# x axis for prediction
def generate_x_axis(start, end, steps):
    return [start + i * (end-start)/(steps-1) for i in range(steps)]

_buffer = 0.25*(max(datapoints.X)-min(datapoints.X))

axis_start, axis_end = (min(datapoints.X) - _buffer, max(datapoints.X) + _buffer)
xaxis = generate_x_axis(axis_start, axis_end, 100)


# gaussian process regression
gp = GaussianProcessRegressionPy(length_scale=0.8, prior_variance=9, data_noise=0)
gp.set_data(datapoints.X, datapoints.Y)
# gp.set_prior_mean(lambda x: 10+10*math.sin(x),xaxis)  # not specifying a prior defaults to constant 0
gp.predict(xaxis)  # stores in gp.prediction


# chart
brythonChart = BrythonChartJS("brythonChart")
brythonChart.datapoints = [datapoints.X, datapoints.Y]
brythonChart.prediction = [xaxis, gp.prediction[0], gp.prediction[1]]
brythonChart.prior = [xaxis, gp.prior[0], gp.prior[1]]

# super hacky but no idea how to do that better atm without polluting the chart class with gp class methods
def pre_update_hook():
    gp.set_data(*brythonChart.datapoints_inputform)
    print(f"Number of Datapoints: {len(brythonChart.datapoints_inputform[0])}")
    gp.predict(xaxis)
    brythonChart.prediction = [xaxis, gp.prediction[0], gp.prediction[1]]
brythonChart.pre_update_hook = pre_update_hook

brythonChart.update_chart()
