namespace GradientDescent
{
    public static class LossFunctions
    {

    }
}
/** testcase    
 * // Example linear model: y = w * x + b, theta = [w, b]
Tensor LinearModel(Tensor xs, Tensor theta)
{
    double w = theta[0];
    double b = theta[1];
    var result = new Tensor(xs.Length);
    for (int i = 0; i < xs.Length; i++)
        result[i] = w * xs[i] + b;
    return result;
}

// Data
var xs = new Tensor(new double[] { 2.0, 1.0, 4.0, 3.0 });
var ys = new Tensor(new double[] { 1.8, 1.2, 4.2, 3.3 });


// Create L2 loss function for this model
var l2Loss = LossFunctions.L2Loss(LinearModel);

// Get the loss function for this data
var lossForData = l2Loss(xs, ys);

// Evaluate loss at theta = [0.0, 0.0]
double loss = lossForData(new Tensor(new double[] { 0.0, 0.0 }));
Console.WriteLine(loss); // Output: 33.21

 ***/