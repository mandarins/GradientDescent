using System;
namespace GradientDescent
{
    public class GradientDescentManager
    {
        private readonly double _alpha; // Learning rate
        private readonly int _revs;     // Number of iterations

        public GradientDescentManager(double alpha, int revs)
        {
            _alpha = alpha;
            _revs = revs;
        }

        /// <summary>
        /// Returns a function that computes the L2 loss for a given model using Tensors.
        /// </summary>
        /// <param name="model">A function that takes input xs and parameters theta, and returns predicted ys as a Tensor.</param>
        /// <returns>A function that takes xs, ys, and returns a function that takes theta and returns the L2 loss.</returns>
        public static Func<Tensor, Tensor, Func<Tensor, double>> L2Loss(
            Func<Tensor, Tensor, Tensor> model)
        {
            return (xs, ys) => (theta) =>
            {
                var predYs = model(xs, theta);
                if (ys.Length != predYs.Length)
                    throw new ArgumentException("Target and prediction tensors must have the same length.");

                double sum = 0.0;
                for (int i = 0; i < ys.Length; i++)
                {
                    double diff = ys[i] - predYs[i];
                    sum += diff * diff;
                }
                return sum;
            };
        }

        /// <summary>
        /// Performs gradient descent to minimize the given objective function.
        /// </summary>
        /// <param name="objective">A function that takes a Tensor theta and returns a scalar loss.</param>
        /// <param name="gradientOf">A function that takes a function and a Tensor, and returns the gradient as a Tensor.</param>
        /// <param name="theta">Initial parameter Tensor.</param>
        /// <returns>The optimized parameter Tensor.</returns>
        public Tensor CalculateGradientDescent(
            Func<Tensor, double> objective,
            Func<Func<Tensor, double>, Tensor, Tensor> gradientOf,
            Tensor theta,
            double alpha,
            int revs)
        {
            // The revision function: performs a single gradient descent step
            Func<Tensor, Tensor> revision = currentTheta =>
            {
                var grad = gradientOf(objective, currentTheta);
                var updated = currentTheta.Clone();
                for (int i = 0; i < updated.Length; i++)
                    updated[i] -= alpha * grad[i];
                return updated;
            };

            // Use the Revise method for iterative updates
            return Revise(revision, revs, theta);
        }
        /// <summary>
        /// Applies the function f to theta, revs times, returning the final result.
        /// </summary>
        /// <typeparam name="T">The type of the parameter being revised (e.g., Tensor, double[], etc.).</typeparam>
        /// <param name="f">The revision function.</param>
        /// <param name="revs">The number of revisions to perform.</param>
        /// <param name="theta">The initial parameter value.</param>
        /// <returns>The revised parameter after revs iterations.</returns>
        public static T Revise<T>(Func<T, T> f, int revs, T theta)
        {
            while (revs > 0)
            {
                theta = f(theta);
                revs--;
            }
            return theta;
        }

        // Numerical gradient (finite difference)
        public static Tensor NumericalGradient(Func<Tensor, double> f, Tensor theta, double eps = 1e-6)
        {
            var grad = new Tensor(theta.Length);
            for (int i = 0; i < theta.Length; i++)
            {
                var thetaPlus = theta.Clone();
                thetaPlus[i] += eps;
                var thetaMinus = theta.Clone();
                thetaMinus[i] -= eps;
                grad[i] = (f(thetaPlus) - f(thetaMinus)) / (2 * eps);
            }
            return grad;
        }

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