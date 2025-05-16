using System;
using System.Drawing;
using System.Linq;
using ScottPlot;

namespace GradientDescent
{
    public static class PlotUtil
    {
        /// <summary>
        /// Plots input data points and a fitted line.
        /// </summary>
        /// <param name="xs">Input x values (Tensor)</param>
        /// <param name="ys">Input y values (Tensor)</param>
        /// <param name="theta">Optimized parameters (Tensor, [w, b])</param>
        /// <param name="title">Plot title</param>
        public static void PlotFit(
                Tensor xs,
                Tensor ys,
                Tensor theta,
                Func<double, Tensor, double> modelFunc,
                string title = "Gradient Descent Plot")
        {
            double[] xData = xs.Data;
            double[] yData = ys.Data;

            double xMin = xData.Min();
            double xMax = xData.Max();
            double[] xLine = new double[] { xMin, xMax, (xMin + xMax) / 2.0 };
            double[] yLine = xLine.Select(x => modelFunc(x, theta)).ToArray();

            var plt = new ScottPlot.Plot();

            plt.Add.ScatterPoints(xData, yData, color: Colors.Blue);
            plt.Add.ScatterPoints(xLine, yLine, color: Colors.Red);

            plt.Title(title);
            plt.XLabel("x");
            plt.YLabel("y");
            plt.SavePng(title + ".png", 400, 300);
        }

        public static void PlotPlaneFit2DProjection(
            Tensor[] xsPlane, Tensor ysPlane, Tensor theta, string title = "Plane Fit 2D Projection")
        {
            // Project x1 vs y for x2 = 0
            double[] x1s = xsPlane.Select(t => t[0]).ToArray();
            double[] x2s = xsPlane.Select(t => t[1]).ToArray();
            double[] ys = ysPlane.Data;

            // Fit line for x2 = 0
            double x1Min = x1s.Min();
            double x1Max = x1s.Max();
            double[] x1Line = new double[] { x1Min, x1Max };
            double[] yLine = x1Line.Select(x1 => theta[0] * x1 + theta[1] * 0 + theta[2]).ToArray();

            var plt = new ScottPlot.Plot();

            // Data points: x1 vs y, colored by x2
            plt.Add.ScatterPoints(x1s, ys, color: Colors.Blue);

            // Plane slice
            plt.Add.ScatterPoints(x1Line, yLine, color: Colors.Red);

            plt.Title(title);
            plt.XLabel("x1 (x2=0)");
            plt.YLabel("y");
            plt.SavePng(title + ".png", 400, 300);
        }

    }
}
