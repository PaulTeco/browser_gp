

// TODO: buttons to change: length scale, sigma, choice of kernel (generally parameters per slider), randomize points, initialize with x points
// even: multiply and add kernels in 2D UI, add vertical, multiply horizontally
// also bind to table? --> angular?


// GP class
class GaussianProcessRegression{
    constructor(variance_point=0){
        this.variance_point = variance_point

        this.dataset_prediction = []
        this.dataset_sigma_low = []
        this.dataset_sigma_high = []

        this.x_axis = []
    }


    RBF = function(xi, xj, length_scale, variance){
        return variance * Math.exp(-((xi-xj)**2)/(2 * length_scale**2))
    }

    // use this function to choose, add and multiply kernel functions. Not implemented yet. 
    set_kernel = function(RBF_length, RBF_variance){
        this.kernel = function(xi, xj){
            return this.RBF(xi, xj, RBF_length, RBF_variance)
        }
    }


    // math.js matrix functions: https://mathjs.org/docs/reference/functions.html#matrix-functions
    // predicts a single mu_t, variance_t for a given x_t
    _train_and_predict = function(X, Y, x_t){

        // computes the covariance matrix for a single x_t with all datapoints X
        var X_new = [...X, x_t]  // add single item to array in copy
        var covariance_matrix = []

        for (let i=0; i<X_new.length; i++){
            covariance_matrix[i]=[]
            for (let j=0; j<X_new.length; j++){
                covariance_matrix[i][j] = this.kernel(X_new[i],X_new[j])
            }
        }

        if (X.length>0){
            // with Sigma being a n+1 x n+1 matrix
            // and Y being a n vector of labels
            var K = math.transpose(math.transpose(covariance_matrix.slice(0,-1)).slice(0,-1))  // original size nxn matrix
            var Kstar = covariance_matrix[covariance_matrix.length-1].slice(0,-1)  // size n vector
            var KstarT = math.transpose(Kstar)
            var Kstarstar = covariance_matrix[covariance_matrix.length-1][covariance_matrix.length-1]  // number

            var Kinv = math.inv(K)  // nxn matrix
            var mu_t = math.multiply(math.multiply(KstarT,Kinv),Y)

            // self variance of measurement points is added here
            var variance_t = Kstarstar - math.multiply(math.multiply(KstarT,Kinv),Kstar) + this.variance_point

        } else {
            var mu_t = 0
            var variance_t = covariance_matrix[0][0] + this.variance_point // get prior dynamically from the input data
        }

            this.dataset_prediction.push({x: x_t, y: mu_t})
            this.dataset_sigma_low.push({x: x_t, y: mu_t-variance_t})
            this.dataset_sigma_high.push({x: x_t, y: mu_t+variance_t})
    }

    generate_x_axis = function(start, end, steps){
        this.x_axis = []
        for (let i=0; i<steps; i++){
            this.x_axis.push(start + i * (end-start)/(steps-1))
        }
    }

    train_gp = function(X, Y){
        this.dataset_prediction = []
        this.dataset_sigma_low = []
        this.dataset_sigma_high = []

        for (let i=0; i<this.x_axis.length; i++){
            this._train_and_predict(X, Y, this.x_axis[i])

        }
    }

}



// Generate Data for usage
class RandomDataPoints{
    constructor(){
        this.X = []
        this.Y = []
        this.XY = []

        for (let i=0; i<10; i++){
    
            if(i==0){
                this.X[i] = 0.2
                this.Y[i] = this.normal_random(10,2)
            } else {
                this.X[i] = this.normal_random(i,0.2)
                this.Y[i] = this.Y[i-1] + this.normal_random(2,5) + 2
            }
        
            this.XY[i] = {x: this.X[i], y: this.Y[i], r:10}
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
    
    update_XY = function(){
        this.XY = []
        for (let i=0; i<this.X.length; i++){
            this.XY[i] = {x: this.X[i], y: this.Y[i], r:10}
        }
    }
}





// actual code start

var datapoints = new RandomDataPoints()

var _length_scale = 0.8
var _sigma_pred_squared = 9
var _sigma_self_squared = 1

var _sigma_squared = _sigma_pred_squared - _sigma_self_squared

var gp = new GaussianProcessRegression(variance_point=1)
gp.set_kernel(RBF_length=_length_scale, RBF_variance=_sigma_squared)
var _buffer = 0.25*(Math.max(...datapoints.X)-Math.min(...datapoints.X))
gp.generate_x_axis(Math.min(...datapoints.X)-_buffer,Math.max(...datapoints.X)+_buffer,100)
gp.train_gp(datapoints.X, datapoints.Y)






const ctx = document.getElementById('pureJSChart').getContext('2d');

// Setting up the Data and the Chart
const chart_data = {
    datasets: [{
        type: 'bubble',
        label: 'Test Dataset',
        data: datapoints.XY,
        backgroundColor: 'rgb(99, 255, 132)'
    },{
        type: 'line',
        label: 'GP Upper Variance',
        fill: '+1',
        data: gp.dataset_sigma_high,
        backgroundColor: 'rgba(132, 99, 255, 0.2)',
        pointRadius: 0,
        pointHitRadius: 5
    },{
        type: 'line',
        label: 'GP Prediction',
        data: gp.dataset_prediction,
        backgroundColor: 'rgba(132, 99, 255, 1)'
    },{
        type: 'line',
        label: 'GP Lower Variance',
        fill: "-1",
        data: gp.dataset_sigma_low,
        backgroundColor: 'rgba(132, 99, 255, 0.2)',
        pointRadius: 0,
        pointHitRadius: 5
    }]
};

const pureJSChart = new Chart(ctx, {
    data: chart_data,
    options: {
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
        datapoints.X.push(new_x)
        datapoints.Y.push(new_y)
    }
    datapoints.update_XY();
    pureJSChart.data.datasets[0].data = datapoints.XY  // update dataset. bit hacky but works for now
    console.log(`Number of Datapoints: ${datapoints.XY.length}`)
    pureJSChart.update(mode="none");  // disables shuffling of points when the array is rearranged
    gp.train_gp(datapoints.X, datapoints.Y);
    pureJSChart.data.datasets[1].data = gp.dataset_sigma_high
    pureJSChart.data.datasets[2].data = gp.dataset_prediction
    pureJSChart.data.datasets[3].data = gp.dataset_sigma_low
    pureJSChart.update();
}