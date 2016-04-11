using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class Positions
    {

        public static List<Vector<double>> ListOfPositionsOnProfile(Domain d, Vector<double> p_1, Vector<double> p_2, int N)
        {
            if (N <= 0) throw new ArgumentException("N is not a positive number");

            List<Vector<double>> list = new List<Vector<double>>(N);

            Vector<double> dr = p_2 - p_1;
            double dx = dr.X / N;
            double dy = dr.Y / N;
            double dz = dr.Z / N;

            for (int i = 0; i < N; i++)
            {
                list.Add(new Vector<double>(i * dx + 0.5 * dx, i * dy + 0.5 * dy, i * dz + 0.5 * dz));
            }

            return list;
        }

        private enum Plane { XY, XZ, YZ };

        public static List<Vector<double>> ListOfPositionOnSlice(Domain domain, Vector<double> p_1, Vector<double> p_2, Vector<double> p_3, int N, int M)
        {
            if ((N <= 0) || (M <= 0)) throw new ArgumentException("N or M is not a positive number");

            if (M > N)
            {
                int help = N;
                N = M;
                M = help;
            }

            List<Vector<double>> list = new List<Vector<double>>(N * M);

            Vector<double> abc = (p_2 - p_1) % (p_3 - p_1);
            double a = abc.X;
            double b = abc.Y;
            double c = abc.Z;
            double d = (a * p_1.X + b * p_1.Y + c * p_1.Z);

            Console.WriteLine("{0}, {1}, {2}, {3}", a, b, c, d);

            double ab = 0.0;
            double ac = 0.0;
            double bc = 0.0;

            if (b != 0.0) ab = Math.Abs(a / b);
            if (c != 0.0) ac = Math.Abs(a / c);
            if (c != 0.0) bc = Math.Abs(b / c);

            Console.WriteLine("{0}, {1}, {2}", ab, ac, bc);

            Plane plane;

            if ((a == 0.0) && (b == 0.0)) plane = Plane.XY;
            else if ((b == 0.0) && (c == 0.0)) plane = Plane.YZ;
            else if ((a == 0.0) && (c == 0.0)) plane = Plane.XZ;
            else
            {
                if (ab <= 1.0) // y
                {
                    if (ac <= 1.0) plane = Plane.YZ;
                    else plane = Plane.XY;
                }
                else // x
                {
                    if (bc <= 1.0) plane = Plane.XZ;
                    else plane = Plane.XY;
                }
            }


            int NX = 0; double dx = 0.0;
            int NY = 0; double dy = 0.0;
            int NZ = 0; double dz = 0.0;

            switch (plane)
            {
                case Plane.XY:
                    if ((int)domain.Parameters["NXC"] > (int)domain.Parameters["NYC"]) { NX = N; NY = M; }
                    else { NY = N; NX = M; }

                    dx = (double)domain.Parameters["XCV"] / NX;
                    dy = (double)domain.Parameters["YCV"] / NY;

                    for (int i = 0; i < NX; i++)
                    {
                        for (int j = 0; j < NY; j++)
                        {
                            double x = i * dx + 0.5 * dx;
                            double y = j * dy + 0.5 * dy;
                            double z = (d - a * x - b * y) / c;
                            if ((z >= 0.0) && (z < (double)domain.Parameters["ZCV"]))
                            {
                                list.Add(new Vector<double>(x, y, z));
                            }
                        }
                    }
                    break;
                case Plane.XZ:
                    if ((int)domain.Parameters["NXC"] > (int)domain.Parameters["NZC"]) { NX = N; NZ = M; }
                    else { NZ = N; NX = M; }

                    dx = (double)domain.Parameters["XCV"] / NX;
                    dz = (double)domain.Parameters["ZCV"] / NZ;

                    for (int i = 0; i < NZ; i++)
                    {
                        for (int j = 0; j < NX; j++)
                        {
                            double z = i * dz + 0.5 * dz;
                            double x = j * dx + 0.5 * dx;
                            double y = (d - a * x - c * z) / b;
                            if ((y >= 0.0) && (y < (double)domain.Parameters["YCV"]))
                            {
                                list.Add(new Vector<double>(x, y, z));
                            }
                            
                        }
                    }
                    break;
                case Plane.YZ:
                    if ((int)domain.Parameters["NYC"] > (int)domain.Parameters["NZC"]) { NY = N; NZ = M; }
                    else { NZ = N; NY = M; }

                    dy = (double)domain.Parameters["YCV"] / NY;
                    dz = (double)domain.Parameters["ZCV"] / NZ;

                    for (int i = 0; i < NZ; i++)
                    {
                        for (int j = 0; j < NY; j++)
                        {
                            double z = i * dz + 0.5 * dz;
                            double y = j * dy + 0.5 * dy;
                            double x = (d - b * y - c * z) / a;
                            if ((x >= 0.0) && (x < (double)domain.Parameters["XCV"]))
                            {
                                list.Add(new Vector<double>(x, y, z));
                            }
                        }
                    }
                    break;
                default:
                    break;
            }

            return list;
        }



        public static List<Vector<double>> ListOfPositionsOnGrid(Domain d, Vector<double> p, double x, double y, double z, int N, int M, int K)
        {
            if ((N <= 0) || (M <= 0) || (K <= 0)) throw new ArgumentException("N or M or K is not a positive number");

            List<Vector<double>> list = new List<Vector<double>>(N * M * K);

            double dx = (x - p.X) / N;
            double dy = (y - p.Y) / M;
            double dz = (z - p.Z) / K;

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        list.Add(new Vector<double>(p.X + i * dx + 0.5 * dx, p.Y + j * dy + 0.5 * dy, p.Z + k * dz + 0.5 * dz));
                    }
                }
            }

            return list;
        }



        public static List<Vector<double>> ListOfPositionsOnGrid(Domain d, Vector<double> p, double x, double y, int N, int M)
        {
            if ((N <= 0) || (M <= 0)) throw new ArgumentException("N or M is not a positive number");

            List<Vector<double>> list = new List<Vector<double>>(N * M);

            double dx = (x - p.X) / N;
            double dy = (y - p.Y) / M;

            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    list.Add(new Vector<double>(p.X + i * dx + 0.5 * dx, p.Y + j * dy + 0.5 * dy, 0.0));
                }
            }

            return list;
        }

    }
}
