using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Sphectacle;

namespace Sphectacle_console
{
    class Program
    {
        [Serializable()]
        public class WrongParametersException : System.Exception
        {
            public WrongParametersException() : base() { }
            public WrongParametersException(string message) : base(message) { }
            public WrongParametersException(string message, System.Exception inner) : base(message, inner) { }

            // A constructor is needed for serialization when an
            // exception propagates from a remoting server to the client. 
            protected WrongParametersException(System.Runtime.Serialization.SerializationInfo info,
                System.Runtime.Serialization.StreamingContext context) { }
        }

        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            Dictionary<string, string> parameters = new Dictionary<string, string>();

            string parameterName = "";
            bool inParameter = false;
            foreach (string arg in args)
            {
                if ( (arg[0] == '-') && (inParameter == false) )
                {
                    if ((arg == "-i") || (arg == "-input")) { parameterName = "input"; inParameter = true; continue; }
                    else if ((arg == "-o") || (arg == "-output")) { parameterName = "output"; inParameter = true; continue; }
                    else if ((arg == "-f") || (arg == "-fields")) { parameterName = "fields"; inParameter = true; continue; }
                    else if ((arg == "-p") || (arg == "-projection")) { parameterName = "projection"; inParameter = true; continue; }
                    else throw new WrongParametersException();
                }
                else if (inParameter == true)
                {
                    if (parameterName == "input")
                    {
                        parameters.Add(parameterName, arg);
                        inParameter = false;
                    }
                    else if (parameterName == "output")
                    {
                        parameters.Add(parameterName, arg);
                        inParameter = false;
                    }
                    else if (parameterName == "fields")
                    {
                        parameters.Add(parameterName, arg);
                        inParameter = false;
                    }
                    else if (parameterName == "projection")
                    {
                        if ((arg == "point") || (arg == "profile") || (arg == "slice") || (arg == "grid"))
                        {
                            
                        }
                        else throw new WrongParametersException();
                    }
                }
                else throw new WrongParametersException();
            }

            if (args.Length == 0) HelpInfo();

            //Domain d = new Domain("C:\\Users\\Kamil Szewc\\Desktop\\16.000000.sph");
            //Domain d = new Domain("C:\\Users\\kamil_000\\Desktop\\16.000000.sph");
            //ParticlesInCells pic = new ParticlesInCells(d);
            //Projector projector = new Projector(pic);

            //List<Vector<double>> positions = Positions.ListOfPositionsOnProfile(d, new Vector<double>(0.0, 0.0, 0.0), new Vector<double>(1.0, 1.0, 1.0), 20);
            //List<Vector<double>> positions = Positions.ListOfPositionsOnGrid(d, new Vector<double>(0.0, 0.0, 0.0), 1.0, 1.0, 1.0,  4, 4, 4);
            //List<Vector<double>> positions = Positions.ListOfPositionOnSlice(d, new Vector<double>(0.5, 0.0, 1.0),
            //                                                                    new Vector<double>(0.0, 0.5, 1.0),
            //                                                                    new Vector<double>(1.0, 1.0, 0.0), 4, 4);

            //Dictionary<string, double[]> res = projector.SphProjectionAtPositions(new string[] { "density" }, positions, Kernels.KernelWendland3D);

            //for (int i = 0; i < positions.Count; i++)
            //{
            //    Console.WriteLine("{0} {1} {2} {3}", positions[i].X, positions[i].Y, positions[i].Z, res["density"][i]);
            //}

            sw.Stop();
            Console.WriteLine(sw.Elapsed);

            Console.ReadKey();
        }


        public static void HelpInfo()
        {
            Console.WriteLine("Help");
        }
    }
}
