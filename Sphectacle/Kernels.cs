using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class Kernels
    {
        public delegate double Kernel(double q, double i_h);

        // Wendland (1995)
        public static double KernelWendland2D(double q, double i_h)
        {
            if (q < 2.0)
                return (1.0/Math.PI) * Math.Pow(i_h, 2) * 0.21875 * Math.Pow(2.0 - q, 4) * (q + 0.5);
            else
                return 0.0;
        }

        public static double GradientOfKernelWendland2D(double x, double q, double i_h)
        {
            if (q < 2.0)
                return -(1.0 / Math.PI) * Math.Pow(i_h, 4) * 1.09375 * x * Math.Pow(2.0 - q, 3);
            else
                return 0.0;
        }

        public static double KernelWendland3D(double q, double i_h)
        {
            if (q < 2.0)
                return (1.0 / Math.PI) * Math.Pow(i_h, 3) * (42.0/256.0) * Math.Pow(2.0 - q, 4) * (q + 0.5);
            else
                return 0.0;
        }

        public static double GradientOfKernelWendland3D(double x, double q, double i_h)
        {
            if (q < 2.0)
                return -(1.0 / Math.PI) * Math.Pow(i_h, 5) * (5.0*42.0/256.0) * x * Math.Pow(2.0 - q, 3);
            else
                return 0.0;
        }

        // Lucy (1977)
        public static double KernelLucy2D(double q, double i_h)
        {
            if (q <= 2.0)
                return (1.0/Math.PI) * Math.Pow(i_h, 2) * (5.0 / 4.0) * (1.0 + 1.5 * q) * Math.Pow(1.0 - 0.5 * q, 3);
            else
                return 0.0;
        }

        public static double GradientOfKernelLucy2D(double x, double q, double i_h)
        {
            if (q <= 2.0)
                return -(1.0 / Math.PI) * Math.Pow(i_h, 4) * (15.0 / 4.0) * Math.Pow(1.0 - 0.5 * q, 2) * x;
            else
                return 0.0;
        }
    }
}
