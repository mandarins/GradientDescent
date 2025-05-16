using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Models should be added in this class
namespace GradientDescent
{
    public class Models
    {
        public Tensor LinearModel(Tensor xs, Tensor theta)
        {
            double w = theta[0];
            double b = theta[1];
            var result = new Tensor(xs.Length);
            for (int i = 0; i < xs.Length; i++)
                result[i] = w * xs[i] + b;
            return result;
        }

        // Quadratic model: y = a * x^2 + b * x + c, theta = [a, b, c]
        public Tensor QuadraticModel(Tensor xs, Tensor theta)
        {
            double a = theta[0];
            double b = theta[1];
            double c = theta[2];
            var result = new Tensor(xs.Length);
            for (int i = 0; i < xs.Length; i++)
                result[i] = a * xs[i] * xs[i] + b * xs[i] + c;
            return result;
        }

        public Tensor PolynomialModel(Tensor xs, Tensor theta)
        {
            var result = new Tensor(xs.Length);
            for (int i = 0; i < xs.Length; i++)
            {
                double y = 0;
                for (int j = 0; j < theta.Length; j++)
                    y += theta[j] * Math.Pow(xs[i], j);
                result[i] = y;
            }
            return result;
        }

        public Tensor PlaneModel(Tensor[] xs, Tensor theta)
        {
            double w1 = theta[0];
            double w2 = theta[1];
            double b = theta[2];
            var result = new Tensor(xs.Length);
            for (int i = 0; i < xs.Length; i++)
            {
                double x1 = xs[i][0];
                double x2 = xs[i][1];
                result[i] = w1 * x1 + w2 * x2 + b;
            }
            return result;
        }

        public Tensor PlaneModelFlat(Tensor xsFlat, Tensor theta)
        {
            int n = xsFlat.Length / 2;
            double w1 = theta[0];
            double w2 = theta[1];
            double b = theta[2];
            var result = new Tensor(n);
            for (int i = 0; i < n; i++)
            {
                double x1 = xsFlat[2 * i];
                double x2 = xsFlat[2 * i + 1];
                result[i] = w1 * x1 + w2 * x2 + b;
            }
            return result;
        }
    }
}


/***** Quadratic sample
 var xs = new Tensor(new double[] { -1.0, 0.0, 1.0, 2.0, 3.0 });
var ys = new Tensor(new double[] { 2.55, 2.1, 4.35, 10.2, 18.25 });

// Loss function
var l2Loss = LossFunctions.L2Loss(QuadraticModel);
var lossForData = l2Loss(xs, ys);

// Initial theta
var theta0 = new Tensor(new double[] { 0.0, 0.0, 0.0 });

// Gradient descent
var gd = new GradientDescent(alpha: 0.001, revs: 1000);
var result = gd.Minimize(
    objective: lossForData,
    gradientOf: GradientDescent.NumericalGradient,
    theta: theta0,
    alpha: 0.001,
    revs: 1000
 * ***********/

/* * 
For a cubic/polynomial model (y ≈ x^3 - x^2 + 2x + 1)
var xsPoly = new Tensor(new double[] { -2, -1, 0, 1, 2 });
var ysPoly = new Tensor(new double[] { -9, -2, 1, 3, 11 });
**/

// Example data: 3 points in 2D
/*
var xs = new Tensor[]
{
    new Tensor(new double[] { 1.0, 2.0 }),
    new Tensor(new double[] { 2.0, 1.0 }),
    new Tensor(new double[] { 3.0, 0.0 })
};
var ys = new Tensor(new double[] { 5.0, 6.0, 7.0 }); // target outputs

// Initial theta: [w1, w2, b]
var theta0 = new Tensor(new double[] { 0.0, 0.0, 0.0 });

// Loss function
var l2Loss = LossFunctions.L2Loss(Models.PlaneModel);
var lossForData = l2Loss(xs, ys);

// Gradient descent
var gd = new GradientDescentManager(alpha: 0.001, revs: 1000);
var result = gd.calculateGradientDescent(
    objective: lossForData,
    gradientOf: GradientDescent.NumericalGradient,
    theta: theta0,
    alpha: 0.001,
    revs: 1000
);
*/