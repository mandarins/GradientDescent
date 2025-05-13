// See https://aka.ms/new-console-template for more information
using GradientDescent;
namespace GradientDescent
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Linear Model Test !");
            double alpha = 0.01;
            int revsVal = 1000;

            // Data
            var xs = new Tensor(new double[] { 2.0, 1.0, 4.0, 3.0 });
            var ys = new Tensor(new double[] { 1.8, 1.2, 4.2, 3.3 });

            var gdm = new GradientDescentManager(alpha, revsVal);
            // Model and loss
            var models = new GradientDescent.Models();
            var l2Loss = GradientDescentManager.L2Loss(models.LinearModel);
            var lossForData = l2Loss(xs, ys);

            // Initial theta
            var theta0 = new Tensor(new double[] { 0.0, 0.0 });


            // Linear model function
            Func<double, Tensor, double> linearModel = (x, t) => t[0] * x + t[1];

            // Call PlotFit for linear
            PlotUtil.PlotFit(xs, ys, theta0, linearModel, "InitialLine");

            // Fixed the issue by explicitly creating a lambda function to match the expected delegate type.
            var result = gdm.CalculateGradientDescent(
                objective: lossForData,
                gradientOf: (f, t) => GradientDescentManager.NumericalGradient(f, t), // Explicit lambda to match Func<Func<Tensor, double>, Tensor, Tensor>
                theta: theta0,
                alpha: alpha,
                revs: revsVal
            );

            PlotUtil.PlotFit(xs, ys, result, linearModel, "ResultingLine");
            Console.WriteLine("Linear Descent theta: " + result);


            var q_xs = new Tensor(new double[] { -1.0, 0.0, 1.0, 2.0, 3.0 });
            var q_ys = new Tensor(new double[] { 2.55, 2.1, 4.35, 10.2, 18.25 });

            // Loss function
            l2Loss = GradientDescentManager.L2Loss(models.QuadraticModel);
            lossForData = l2Loss(q_xs, q_ys);

            // Initial theta
            var q_theta0 = new Tensor(new double[] { 0.0, 0.0, 0.0 });
            Func<double, Tensor, double> quadraticModel = (x, t) => t[0] * x * x + t[1] * x + t[2];
            
            PlotUtil.PlotFit(q_xs, q_ys, q_theta0, quadraticModel, "InitialQuadraticPlot");
            // Gradient descent
            gdm = new GradientDescentManager(alpha: 0.001, revs: 1000);
            var q_result = gdm.CalculateGradientDescent(
                objective: lossForData,
                gradientOf: (f, t) => GradientDescentManager.NumericalGradient(f, t),
                theta: q_theta0,
                alpha: 0.001,
                revs: 1000
            );

            PlotUtil.PlotFit(q_xs, q_ys, q_result, quadraticModel, "ResultingQuadraticPlot");
            Console.WriteLine("Quadratic Descent theta: " + result);
        }
    }
}