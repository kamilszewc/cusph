using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{

    public class Projector
    {
        private ParticlesInCells _particlesInCells;
        private Domain _domain;
        private Parameters _parameters;

        public Projector(ParticlesInCells particlesInCells)
        {
            _particlesInCells = particlesInCells;
            _domain = _particlesInCells.Domain;
            _parameters = _domain.Parameters;
        }

        public double SphProjectionAtPosition(string name, Vector<double> position, Kernels.Kernel kernel)
        {
            List<Particle> particles = _particlesInCells.GetListOfParticlesNeighbouringToPosition(position);

            List<Particle> ghostParticles = _particlesInCells.GetListOfGhostParticlesNeighbouringToPosition(position);
            particles.AddRange(ghostParticles);

            double result = 0.0;

            foreach (Particle particle in particles)
            {
                double r = Math.Sqrt( Math.Pow(particle.Position.X - position.X, 2)
                                    + Math.Pow(particle.Position.Y - position.Y, 2)
                                    + Math.Pow(particle.Position.Z - position.Z, 2) );

                double q = r * _parameters.I_H;

                if (q < 2.0)
                {
                    result += (dynamic)particle[name] * kernel(q, _parameters.I_H) * (double)particle["mass"] / (double)particle["density"];
                }
            }

            return result;
        }

        public Dictionary<string, double> SphProjectionAtPosition(string[] names, Vector<double> position, Kernels.Kernel kernel)
        {
            List<Particle> particles = _particlesInCells.GetListOfParticlesNeighbouringToPosition(position);

            List<Particle> ghostParticles = _particlesInCells.GetListOfGhostParticlesNeighbouringToPosition(position);
            particles.AddRange(ghostParticles);

            double[] results = (from name in names select 0.0).ToArray();

            foreach (Particle particle in particles)
            {
                double r = Math.Sqrt(Math.Pow(particle.Position.X - position.X, 2)
                                    + Math.Pow(particle.Position.Y - position.Y, 2)
                                    + Math.Pow(particle.Position.Z - position.Z, 2));

                double q = r * _parameters.I_H;

                if (q < 2.0)
                {
                    double norm = kernel(q, _parameters.I_H) * (double)particle["mass"] / (double)particle["density"];

                    for (int i = 0; i < names.Length; i++)
                    {
                        results[i] += (dynamic)particle[names[i]] * norm;
                    }
                }
            }

            Dictionary<string, double> dictionary = new Dictionary<string,double>(names.Length);
            for (int i=0; i<names.Length; i++)
            {
                dictionary.Add(names[i], results[i]);
            }

            return dictionary;
        }


        public Dictionary<string, double[]> SphProjectionAtPositions(string[] names, Vector<double>[] positions, Kernels.Kernel kernel)
        {
            Dictionary<string, double[]> dictionary = new Dictionary<string, double[]>(names.Length);

            foreach (var name in names)
            {
                dictionary.Add(name, new double[positions.Length]);
            }

            Parallel.For(0, positions.Length, index =>
                {
                    Dictionary<string, double> fieldValuesAtPoint = SphProjectionAtPosition(names, positions[index], kernel);
                    foreach (string name in names)
                    {
                        double[] fieldValues = dictionary[name];
                        fieldValues[index] = fieldValuesAtPoint[name];
                    }
                }
            );

            return dictionary;
        }

        public Dictionary<string, double[]> SphProjectionAtPositions(string[] names, List<Vector<double>> positions, Kernels.Kernel kernel)
        {
            return SphProjectionAtPositions(names, positions.ToArray(), kernel);
        }
    }
}
