import random
import math
from browser import document, window  # type: ignore

# import chartjs module
chartjs = window.Chart
# import mathjs module
mathjs = window.math


# self._length_scale = 0.8
# self._sigma_pred_squared = 9
# self._sigma_self_squared = 1

class GaussianProcessRegressionPy:
    def __init__(self, length_scale=1, prediction_variance=1, self_variance=0):
        self._length_scale = length_scale
        self._sigma_pred_squared = prediction_variance
        self._sigma_self_squared = self_variance

        self.data_X = None
        self.data_Y = None

        self.prediction = []

    @property
    def _sigma_squared(self):    
        return self._sigma_pred_squared - self._sigma_self_squared

    def _k_RBF(self, xi,xj):
        return(self._sigma_squared * math.exp(-(xi-xj)**2/(2 * self._length_scale**2)))

    def _calc_Sigma(self, x_t):
        X_new = self.data_X + [x_t]
        return [[self._k_RBF(xi,xj) for xj in X_new] for xi in X_new]


    def _calc_gp(self, Sigma):
        if self.data_X:
            # sigma is n+1 x n+1, Y is n vector
            K = mathjs.transpose(mathjs.transpose(Sigma[:-1])[:-1])  # nxn matrix

            Kstar = Sigma[-1][:-1]  # n vector
            Kstarstar = Sigma[-1][-1]  # scalar

            KstarT = list(Kstar)  # n vector copy
            Kinv = mathjs.inv(K)  # nxn matrix --> need javascript mathjs library for inversion

            mu_t = mathjs.multiply(mathjs.multiply(KstarT,Kinv),self.data_Y)
            variance_t = Kstarstar - mathjs.multiply(mathjs.multiply(KstarT,Kinv),Kstar) + self._sigma_self_squared
        else:
            mu_t = 0
            variance_t = Sigma[0][0] + self._sigma_self_squared
        return [mu_t, variance_t]

    def set_data(self, X, Y):
        self.data_X = X
        self.data_Y = Y

    def predict(self, x_axis):
        self.prediction = [self._calc_gp(self._calc_Sigma(x_pred)) for x_pred in x_axis]




# chartjs 
class BrythonChartJS:
    def __init__(self, div_id):

        self._datapoints = []  # input: [[x1, x2, x3, ..., xn], [y1, y2, y3, ..., yn]] --> [{"x": x1, "y": y1, "r": 10}, ..., {"x": xn, "y": yn, "r": 10}]
        self._prediction = []  # input: [[x1, x2, x3, ..., xk], [[mu1, var1], [mu2, var2], ..., [muk, vark]]] --> [{"x": x1, "y": y1}, ..., {"x": xn, "y": yn}]
        self._prediction_add_variance = []
        self._prediction_sub_variance = []

        self._chart = chartjs.new(document[div_id].getContext('2d'),{
            "data": {
                "datasets": [{
                    "type": 'bubble',
                    "label": 'Dataset',
                    "data": self._datapoints,
                    "backgroundColor": 'rgb(99, 255, 132)'
                },{
                    "type": 'line',
                    "label": 'GP Upper Variance',
                    "fill": '+1',
                    "data": self._prediction_add_variance,
                    "backgroundColor": 'rgba(132, 99, 255, 0.2)',
                    "pointRadius": 0,
                    "pointHitRadius": 5
                },{
                    "type": 'line',
                    "label": 'GP Prediction',
                    "data": self._prediction,
                    "backgroundColor": 'rgba(132, 99, 255, 1)'
                },{
                    "type": 'line',
                    "label": 'GP Lower Variance',
                    "fill": "-1",
                    "data": self._prediction_sub_variance,
                    "backgroundColor": 'rgba(132, 99, 255, 0.2)',
                    "pointRadius": 0,
                    "pointHitRadius": 5
                }]
            },
            "options": {
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
            }
        })

        self.pre_update_hook = None

    @property
    def datapoints(self):
        return self._datapoints

    # setter method automatically transforms into chart version (and updates gp prediction?)
    @datapoints.setter
    def datapoints(self, val):
        inputT = [[val[j][i] for j in range(len(val))] for i in range(len(val[0]))]
        self._datapoints = [{"x": el[0], "y": el[1], "r": 10} for el in inputT]

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, val):
        x, mu_var = val
        mu, var = [[mu_var[j][i] for j in range(len(mu_var))] for i in range(len(mu_var[0]))]
        x_mu_var_T = list(zip(x, mu, var))

        self._prediction = [{"x": el[0], "y": el[1]} for el in x_mu_var_T]
        self._prediction_add_variance = [{"x": el[0], "y": el[1] + el[2]} for el in x_mu_var_T]
        self._prediction_sub_variance = [{"x": el[0], "y": el[1] - el[2]} for el in x_mu_var_T]

    # used to train the gp when points have been removed or added interactively
    @property
    def datapoints_inputform(self):
        return [[el["x"] for el in self._datapoints],[el["y"] for el in self._datapoints]]

    def add_datapoint(self, data):  # takes data as [x, y]
        self._datapoints += [{"x": data[0], "y": data[1], "r": 10}]

    def update_chart(self):
        for ds in self._chart["data"]["datasets"]:
            if ds["label"] == "Dataset":
                ds["data"] = self._datapoints
                self._chart.update("none")
            elif ds["label"] == "GP Upper Variance":
                ds["data"] = self._prediction_add_variance
            elif ds["label"] == "GP Prediction":
                ds["data"] = self._prediction
            elif ds["label"] == "GP Lower Variance":
                ds["data"] = self._prediction_sub_variance

            self._chart.update()

    # args catches whatever else is handed over
    def click_chart(self, event, *args):
        element = self._chart.getElementsAtEventForMode(event, 'point', self._chart["options"])
        if len(element) > 0:
            for el in element:
                if el.datasetIndex == 0:
                    self._datapoints.pop(el.index)


        else:
            canvasPosition = chartjs.helpers.getRelativePosition(event, self._chart)
            new_xy = [self._chart.scales.x.getValueForPixel(canvasPosition.x), self._chart.scales.y.getValueForPixel(canvasPosition.y)]
            self.add_datapoint(new_xy)

        if self.pre_update_hook is not None:
            self.pre_update_hook()
        self.update_chart()



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
gp = GaussianProcessRegressionPy(length_scale=0.8, prediction_variance=9, self_variance=1)
gp.set_data(datapoints.X, datapoints.Y)
gp.predict(xaxis)  # stores in gp.prediction


# chart
brythonChart = BrythonChartJS("brythonChart")
brythonChart.datapoints = [datapoints.X, datapoints.Y]
brythonChart.prediction = [xaxis, gp.prediction]

# super hacky but no idea how to do that better atm without polluting the chart class with gp class methods
def pre_update_hook():
    gp.set_data(*brythonChart.datapoints_inputform)
    print(f"Number of Datapoints: {len(brythonChart.datapoints_inputform[0])}")
    gp.predict(xaxis)
    brythonChart.prediction = [xaxis, gp.prediction]
brythonChart.pre_update_hook = pre_update_hook

brythonChart.update_chart()
