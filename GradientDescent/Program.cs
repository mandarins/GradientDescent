// See https://aka.ms/new-console-template for more information
using GradientDescent;
namespace GradientDescent
{
    class Program
    {


        private static readonly string LogFilePath = "gradientdescent.log";

        public static void LogMessage(string message)
        {
            // Appends the message to the log file with a newline
            File.AppendAllText(LogFilePath, message + Environment.NewLine);
        }

        static void Main(string[] args)
        {
            var pm = new Program();

            Console.WriteLine("Linear Model Test !");
            Program.LogMessage("Linear Model Test !");


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
                gradientOf: (f, t) => GradientDescentManager.CalcutateDelta(f, t), // Explicit lambda to match Func<Func<Tensor, double>, Tensor, Tensor>
                theta: theta0,
                alpha: alpha,
                revs: revsVal
            );

            PlotUtil.PlotFit(xs, ys, result, linearModel, "ResultingLine");
            Console.WriteLine("Linear Descent theta: " + result);
            Program.LogMessage("Linear Descent theta: " + result);  


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
                gradientOf: (f, t) => GradientDescentManager.CalcutateDelta(f, t),
                theta: q_theta0,
                alpha: 0.001,
                revs: 1000
            );

            PlotUtil.PlotFit(q_xs, q_ys, q_result, quadraticModel, "ResultingQuadraticPlot");
            Console.WriteLine("Quadratic Descent theta: " + result);
            Program.LogMessage("Quadratic Descent theta: " + result);


            /****** POLY NOMIAL TEST *********/
            // For a cubic / polynomial model(y ≈ x ^ 3 - x ^ 2 + 2x + 1)
            //var poly_xs = new Tensor(new double[] { -2, -1, 0, 1, 2 });
            //var poly_ys = new Tensor(new double[] { -9, -2, 1, 3, 11 });
            double[] xVals = Enumerable.Range(0, 21).Select(i => -3.0 + i * 0.3).ToArray();
            double[] yVals = xVals.Select(x => Math.Sin(x)).ToArray();

            var poly_xs = new Tensor(xVals);
            var poly_ys = new Tensor(yVals);
            // Loss function
            l2Loss = GradientDescentManager.L2Loss(models.PolynomialModel);
            lossForData = l2Loss(poly_xs, poly_ys);


            // For a cubic (degree 3) polynomial: y = a*x^3 + b*x^2 + c*x + d, theta = [d, c, b, a]
            //var poly_theta0 = new Tensor(new double[] { 0.0, 0.0, 0.0, 0.0 }); // 4 coefficients, all start at 0
            var poly_theta0 = new Tensor(new double[] { .01, .01, .01, .01 }); // 4 coefficients, all start at 0


            Func<double, Tensor, double> polyPlotterModel = (x, t) =>
            {
                double y = 0;
                for (int i = 0; i < t.Length; i++)
                    y += t[i] * Math.Pow(x, i);
                return y;
            };

            PlotUtil.PlotFit(poly_xs, poly_ys, poly_theta0, polyPlotterModel, "InitialPolyNomialPlotting");

            // Gradient descent
            gdm = new GradientDescentManager(alpha: 0.001, revs: 1000);
            var poly_result = gdm.CalculateGradientDescent(
                objective: lossForData,
                gradientOf: (f, t) => GradientDescentManager.CalcutateDelta(f, t),
                theta: poly_theta0,
                alpha: 0.001,
                revs: 1000
            );

            PlotUtil.PlotFit(poly_xs, poly_ys, poly_result, polyPlotterModel, "ResultPolyNomialPlotting");
            Program.LogMessage("Polynomial Descent theta: " + poly_result);

            //******** TEST A PLANE MODEL *************
            var xsPlane = new Tensor[]
            {
                new Tensor(new double[] { 1.0, 2.0 }),
                new Tensor(new double[] { 2.0, 1.0 }),
                new Tensor(new double[] { 3.0, 0.0 })
            };
            var ysPlane = new Tensor(new double[] { 5.0, 6.0, 7.0 }); // target outputs

            // Initial theta: [w1, w2, b]
            var thetaPlane0 = new Tensor(new double[] { 0.0, 0.0, 0.0 });

            // Fix for CS1503: Adjust the lambda function to correctly handle the input type 'Tensor[]' for the PlaneModel.
            var l2LossPlane = GradientDescentManager.L2LossForTensorArray(models.PlaneModel);
            var lossForDataPlane = l2LossPlane(xsPlane, ysPlane);

            // Gradient descent
            var gdPlane = new GradientDescentManager(alpha: 0.001, revs: 1000);
            var resultPlane = gdPlane.CalculateGradientDescent(
                objective: lossForDataPlane, // <-- Use the correct loss function for the plane model
                gradientOf: (f, t) => GradientDescentManager.CalcutateDelta(f, t),
                theta: thetaPlane0,
                alpha: 0.001,
                revs: 1000
            );

            PlotUtil.PlotPlaneFit2DProjection(xsPlane, ysPlane, resultPlane, "PlaneModel2DProjection");
            Console.WriteLine("Plane Descent theta: " + resultPlane);
            Program.LogMessage("Plane Descent theta: " + resultPlane);

        }


    }
}