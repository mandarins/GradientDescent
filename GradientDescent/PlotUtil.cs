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

    }
}
