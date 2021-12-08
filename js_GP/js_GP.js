// GP class
class GaussianProcessRegression{
    constructor(length_scale=1, prior_variance=1, data_noise=0){
        this._length_scale = length_scale
        this._sigma_prior_squared = prior_variance
        this._sigma_noise_squared = data_noise

        this.dataset_prediction = []
        this.dataset_sigma_low = []
        this.dataset_sigma_high = []
        this.prior_mean = undefined
        this.prior_mean_function = undefined

        this._prior = []
        this._prior_add_variance = []
        this._prior_sub_variance = []
    }

    _k_RBF = function(xi, xj){
        let noise = xi==xj ? this._sigma_noise_squared : 0
        return (this._sigma_prior_squared - this._sigma_noise_squared) * Math.exp(-((xi-xj)**2)/(2 * this._length_scale**2)) + noise
    }

    set_prior_mean = function(func, X_t){
        this.prior_mean_function = func

        this.prior_mean = func(X_t)

        this._prior = this.prior_mean.map((el, idx) => {return {x: X_t[idx], y: el}})
        this._prior_add_variance = this.prior_mean.map((el, idx) => {return {x: X_t[idx], y: el+this._sigma_prior_squared}})
        this._prior_sub_variance = this.prior_mean.map((el, idx) => {return {x: X_t[idx], y: el-this._sigma_prior_squared}})
    }

    // math.js matrix functions: https://mathjs.org/docs/reference/functions.html#matrix-functions
    // predicts arrays mu_t, variance_t for given array X_t
    predict = function(X, Y, X_t){

        // computes the covariance matrix for m prediction points X_t from n datapoints X
        var X_new = [...X, ...X_t]

        const n = X.length
        const m = X_t.length

        var covariance_matrix = []

        for (let i=0; i<n+m; i++){
            covariance_matrix[i]=[]
            for (let j=0; j<n+m; j++){
                covariance_matrix[i][j] = this._k_RBF(X_new[i],X_new[j])
            }
        }

        // set prior to constant 0 if nothing was specified
        if (this.prior_mean === undefined){
            this.set_prior_mean((arr) => {return arr.map((el) => {return 0})}, X_t)
        }

        var mu_t = []
        var variance_t = []

        if (n>0){
            // with Sigma being a n+m x n+m matrix
            // and Y being a n vector of labels
            const K = covariance_matrix.slice(0,n).map(row => row.slice(0,n));  // nxn matrix
            const Kstar = covariance_matrix.slice(0,n).map(row => row.slice(n,n+m));  // nxm matrix
            const KstarT = covariance_matrix.slice(n,n+m).map(row => row.slice(0,n));  // mxn matrix
            const Kstarstar = covariance_matrix.slice(n,n+m).map(row => row.slice(n,n+m));  // mxm matrix

            const Kinv = math.inv(K)  // nxn matrix

            // subtract prior from support points
            const prior_at_X = this.prior_mean_function(X)
            const Y_prior = prior_at_X.map((el, idx) => {return (Y[idx] - el)})  

            const KstarTKinvY = math.multiply(math.multiply(KstarT,Kinv),Y_prior)
            const KstarTKinvKstar = math.multiply(math.multiply(KstarT,Kinv),Kstar)

            for (let i=0; i<m; i++){
                mu_t[i] = KstarTKinvY[i] + this.prior_mean[i]
                variance_t[i] = Kstarstar[i][i] - KstarTKinvKstar[i][i]
            }
            

        } else {

            for (let i=0; i<m; i++){
                mu_t[i] = this.prior_mean[i]
                variance_t[i] = covariance_matrix[i][i]
            }
    
        }
        this.dataset_prediction = []
        this.dataset_sigma_low = []
        this.dataset_sigma_high = []

        for (let i=0; i<m; i++){
            this.dataset_prediction.push({x: X_t[i], y: mu_t[i]})
            this.dataset_sigma_low.push({x: X_t[i], y: mu_t[i] - variance_t[i]})
            this.dataset_sigma_high.push({x: X_t[i], y: mu_t[i] + variance_t[i]})
        }

    }

}



// Generate Data for usage
class RandomDataPoints{
    constructor(){
        this.X = []
        this.Y = []

        for (let i=0; i<10; i++){
            if(i==0){
                this.X[i] = 0.2
                this.Y[i] = this.normal_random(10,2)
            } else {
                this.X[i] = this.normal_random(i,0.2)
                this.Y[i] = this.Y[i-1] + this.normal_random(2,5) + 2
            }
        }
    }

    // adding 6 uniform distributions is gaussian enough for this purpose (central limit theorem)
    normal_random = function(mu,sigma) {
    
        var rand = 0;
      
        for (let i=0; i<6; i++) {
          rand += Math.random();
        }
    
        rand= ((((rand/6) -0.5 ) *2) *5*sigma) +mu
      
        return rand;
    }
    
    // getter for X and Y in one array in chartjs data form
    get XY(){
        return this.X.map((val, idx) => {return {x: val, y: this.Y[idx], r: 10}})
    }
}





// actual code start

function generate_x_axis(start, end, steps){
    var x_axis = []
    for (let i=0; i<steps; i++){
        x_axis.push(start + i * (end-start)/(steps-1))
    }
    return x_axis
}


var datapoints = new RandomDataPoints()

var _buffer = 0.25*(Math.max(...datapoints.X)-Math.min(...datapoints.X))
var xaxis = generate_x_axis(Math.min(...datapoints.X)-_buffer,Math.max(...datapoints.X)+_buffer,100)

var gp = new GaussianProcessRegression(length_scale=0.8, prior_variance=9, data_noise=0)
// gp.set_prior_mean((arr) => {return arr.map((el) => {return el*5})}, xaxis)  // set a custom prior through a function
gp.predict(datapoints.X, datapoints.Y, xaxis)

const ctx = document.getElementById('pureJSChart').getContext('2d');

// Setting up the Data and the Chart
const chart_data = {
    datasets: [{
        type: 'bubble',
        label: 'Dataset',
        data: datapoints.XY,
        backgroundColor: 'rgb(99, 255, 132)'
    },{
        type: 'line',
        label: 'GP Prediction',
        data: gp.dataset_prediction,
        backgroundColor: 'rgba(132, 99, 255, 1)'
    },{
        type: 'line',
        label: 'GP Variance',
        fill: '+1',
        data: gp.dataset_sigma_high,
        backgroundColor: 'rgba(132, 99, 255, 0.2)',
        pointRadius: 0,
        pointHitRadius: 5
    },{
        type: 'line',
        label: 'GP Variance',
        fill: "-1",
        data: gp.dataset_sigma_low,
        backgroundColor: 'rgba(132, 99, 255, 0.2)',
        pointRadius: 0,
        pointHitRadius: 5
    },{
        type: 'line',
        label: 'Prior',
        data: gp._prior,
        backgroundColor: 'rgba(32, 99, 255, 0.3)'
    },{
        type: 'line',
        label: 'Prior Variance',
        fill: '+1',
        data: gp._prior_add_variance,
        backgroundColor: 'rgba(32, 99, 255, 0.1)',
        pointRadius: 0,
        pointHitRadius: 5
    },{
        type: 'line',
        label: 'Prior Variance',
        fill: "-1",
        data: gp._prior_sub_variance,
        backgroundColor: 'rgba(32, 99, 255, 0.1)',
        pointRadius: 0,
        pointHitRadius: 5
    }]
};

const pureJSChart = new Chart(ctx, {
    data: chart_data,
    options: {
        plugins: {
            legend: {
                onClick: newLegendClickHandler,
                labels: {
                    filter: newLegendFilterHandler, 
                }
            }
        },
        scales: {
            y: {
                suggestedMax: 50,
                suggestedMin: 0,
            },
            x: {
                suggestedMin: 0,
                suggestedMax: 10
            }
        },
        responsive: true,
        onClick: (e) => {
            chart_onclick(e)
        }
    }
});

// Click Handler from https://www.chartjs.org/docs/3.6.2/configuration/legend.html#legend-item-interface
function newLegendClickHandler(e, legendItem, legend) {

    const index = legendItem.datasetIndex;

    // prediction variance
    if (index == 2 | index == 3) {
        let ci = legend.chart;
        [ci.getDatasetMeta(2), ci.getDatasetMeta(3)].forEach(function(meta) {
            meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
        });
        ci.update();
    } else if (index == 4 | index == 5 | index == 6){
        let ci = legend.chart;
        [ci.getDatasetMeta(4), ci.getDatasetMeta(5), ci.getDatasetMeta(6)].forEach(function(meta) {
            meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
        });
        ci.update();
    } else {
        // Do the original logic
        Chart.defaults.plugins.legend.onClick(e, legendItem, legend);
    }
};
// -------------------------------------------


function newLegendFilterHandler(item, chart){
    if (item.datasetIndex == 3 | item.datasetIndex == 5 | item.datasetIndex == 6){
        return false
    } else {
        return true
    }
}




function chart_onclick(e){
    var element = pureJSChart.getElementsAtEventForMode(e, 'point', pureJSChart.options);
    if(element.length > 0){
        element.forEach(el => {
            if (el.datasetIndex == 0){
                datapoints.X.splice(el.index, 1);
                datapoints.Y.splice(el.index, 1);
            }
        });
    } else {
        const canvasPosition = Chart.helpers.getRelativePosition(e, pureJSChart);
        const new_x = pureJSChart.scales.x.getValueForPixel(canvasPosition.x)
        const new_y = pureJSChart.scales.y.getValueForPixel(canvasPosition.y)

        if (datapoints.X.includes(new_x)){
            datapoints.Y[datapoints.X.indexOf(new_x)] = new_y
        } else {
            datapoints.X.push(new_x)
            datapoints.Y.push(new_y)
        }
    }
    pureJSChart.data.datasets[0].data = datapoints.XY  // update dataset
    console.log(`Number of Datapoints: ${datapoints.XY.length}`)
    pureJSChart.update(mode="none");  // disables shuffling of points when the array is rearranged
    gp.predict(datapoints.X, datapoints.Y, xaxis);
    pureJSChart.data.datasets[1].data = gp.dataset_prediction
    pureJSChart.data.datasets[2].data = gp.dataset_sigma_high
    pureJSChart.data.datasets[3].data = gp.dataset_sigma_low
    // prior and prior variance doesnt need to be added since they dont change on any onclick event
    pureJSChart.update();
}