using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class ParticlesInCells
    {
        private Domain _domain;
        private Parameters _parameters;
        private Particles _particles;
        private int[] _index;
        private int[] _start;
        private int[] _end;

        public Domain Domain { get { return _domain; } }

        public ParticlesInCells(Domain domain)
        {
            _domain = domain;
            _parameters = domain.Parameters;
            _particles = domain.Particles;
            
            int numberOfCells = _domain.NumberOfCells;

            _index = new int[_parameters.N];
            _start = new int[numberOfCells];
            _end = new int[numberOfCells];
            int[] cellHash = new int[_parameters.N];

            for (int i=0; i<_parameters.N; i++)
            {
                cellHash[i] = GetCellHashFromParticleIndex(i);
                _index[i] = i;
            }

            Array.Sort(cellHash, _index);

            int prevHash = -1;
            for (int i=0; i<_parameters.N; i++)
            {
                int nextHash = cellHash[i];
                if (prevHash != nextHash)
                {
                    _start[nextHash] = i;
                }
                _end[nextHash] = i + 1;
                prevHash = nextHash;
            }

        }

        public Vector<int> GetCellPositionFromPosition(Vector<double> position)
        {
            int x = (int)(0.5 * position.X * (double)_parameters.I_H);
            int y = (int)(0.5 * position.Y * (double)_parameters.I_H);
            int z = 0;
            if (_domain.Dimensions == 3) z = (int)(0.5 * position.Z * (double)_parameters.I_H);

            return new Vector<int>(x, y, z);
        }

        public int GetCellHashFromPosition(Vector<double> position)
        {
            Vector<int> cellPosition = GetCellPositionFromPosition(position);
            return GetCellHashFromCellPosition(cellPosition);
        }


        public Vector<int> GetCellPositionFromParticleIndex(int index)
        {
            int x = (int)(0.5 * _particles[index].Position.X * (double)_parameters.I_H);
            int y = (int)(0.5 * _particles[index].Position.Y * (double)_parameters.I_H);
            int z = 0;
            if (_domain.Dimensions == 3) z = (int)(0.5 * _particles[index].Position.Z * (double)_parameters.I_H);

            return new Vector<int>(x, y, z);
        }


        public int GetCellHashFromParticleIndex(int index)
        {
            Vector<int> position = GetCellPositionFromParticleIndex(index);
            return GetCellHashFromCellPosition(position);
        }


        public Vector<int> GetCellPositionFromCellHash(int hash)
        {
            if (_domain.Dimensions == 3)
            {
                int z = hash / ((int)_parameters["NXC"] * (int)_parameters["NYC"]);
                int y = hash % ((int)_parameters["NXC"] * (int)_parameters["NYC"]) / (int)_parameters["NXC"];
                int x = (hash % ((int)_parameters["NXC"] * (int)_parameters["NYC"])) % (int)_parameters["NXC"];
                return new Vector<int>(x, y, z);
            }
            else
            {
                int y = hash / (int)_parameters["NXC"];
                int x = hash % (int)_parameters["NYC"];
                return new Vector<int>(x, y, 0);
            }
        }


        public int GetCellHashFromCellPosition(Vector<int> position)
        {
            if (_domain.Dimensions == 3)
            {
                return position.Z * (int)_parameters["NXC"] * (int)_parameters["NYC"] + position.Y * (int)_parameters["NXC"] + position.X;
            }
            else
            {
                return position.Y * (int)_parameters["NXC"] + position.X;
            }
        }

        public List<Particle> GetListOfParticlesAdjacentToCellHash(int hash)
        {
            List<Particle> particles = new List<Particle>();

            for (int i = _start[hash]; i < _end[hash]; i++)
            {
                particles.Add(_particles.GetParticle(_index[i]));
            }

            return particles;
        }

        public List<Particle> GetListOfParticlesAdjacentToCellPosition(Vector<int> cellPosition)
        {
            int hash = GetCellHashFromCellPosition(cellPosition);
            return GetListOfParticlesAdjacentToCellHash(hash);
        }

        public List<Particle> GetListOfParticlesAdjacentToPosition(Vector<double> position)
        {
            int hash = GetCellHashFromPosition(position);
            return GetListOfParticlesAdjacentToCellHash(hash);
        }

        public List<Particle> GetListOfParticlesNeighbouringToCellPosition(Vector<int> cellPosition)
        {
            List<Particle> particles = new List<Particle>();

            if (_domain.Dimensions == 3)
            {
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        for (int z = -1; z <= 1; z++)
                        {
                            Vector<int> position = new Vector<int>(cellPosition.X + x, cellPosition.Y + y, cellPosition.Z + z);
                            if ((position.X >= 0) && (position.X < (int)_parameters["NXC"])
                              && (position.Y >= 0) && (position.Y < (int)_parameters["NYC"])
                              && (position.Z >= 0) && (position.Z < (int)_parameters["NZC"]))
                            {
                                particles.AddRange(GetListOfParticlesAdjacentToCellPosition(position));
                            }
                        }
                    }
                }
            }
            else
            {
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                            Vector<int> position = new Vector<int>(cellPosition.X + x, cellPosition.Y + y, 0);
                            if ((position.X >= 0) && (position.X < (int)_parameters["NXC"])
                              && (position.Y >= 0) && (position.Y < (int)_parameters["NYC"]) )
                            {
                                particles.AddRange(GetListOfParticlesAdjacentToCellPosition(position));
                            }
                    }
                }
            }

            return particles;
        }
        

        public List<Particle> GetListOfParticlesNeighbouringToPosition(Vector<double> position)
        {
            Vector<int> cellPosition = GetCellPositionFromPosition(position);
            return GetListOfParticlesNeighbouringToCellPosition(cellPosition);
        }

        public List<Particle> GetListOfGhostParticlesNeighbouringToCellPosition(Vector<int> cellPosition)
        {
            List<Particle> ghostParticles = new List<Particle>();

            if (_domain.Dimensions == 3)
            {
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        for (int z = -1; z <= 1; z++)
                        {
                            Vector<int> position = new Vector<int>(cellPosition.X + x, cellPosition.Y + y, cellPosition.Z + z);

                            if ((position.X < 0) && (position.Y < 0) &&  (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, 0, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Y >= (int)_parameters["NYC"]) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, (int)_parameters["NYC"]-1, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = 2.0*(double)_parameters["YCV"] - particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Y < 0) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, 0, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"]  - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Y >= (int)_parameters["NYC"]) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, (int)_parameters["NYC"] - 1, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y < 0) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, 0, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y >= (int)_parameters["NYC"]) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"]-1, (int)_parameters["NYC"] - 1, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y < 0) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, 0, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y >= (int)_parameters["NYC"]) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, (int)_parameters["NYC"] - 1, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Y < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, 0, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Y >= (int)_parameters["NYC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, (int)_parameters["NYC"] - 1, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, position.Y, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X < 0) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, position.Y, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"]-1, 0, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = -particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Y >= (int)_parameters["NYC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, (int)_parameters["NYC"] - 1, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, position.Y, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.X >= (int)_parameters["NXC"]) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, position.Y, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }

                            else if ((position.Y < 0) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, 0, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.Y >= (int)_parameters["NYC"]) && (position.Z < 0))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, (int)_parameters["NYC"]-1, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"]-particle.Position.Y;
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.Y < 0) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, 0, (int)_parameters["NZC"]-1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = -particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if ((position.Y >= (int)_parameters["NYC"]) && (position.Z >= (int)_parameters["NZC"]))
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, (int)_parameters["NYC"]-1, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }

                            else if (position.X < 0)
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, position.Y, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = -particle.Position.X;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if (position.Y < 0)
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, 0, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = -particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if (position.Z < 0)
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, position.Y, 0)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Z = -particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }

                            else if (position.X >= (int)_parameters["NXC"])
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"]-1, position.Y, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.X = 2.0*(double)_parameters["XCV"]-particle.Position.X;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if (position.Y >= (int)_parameters["NYC"])
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, (int)_parameters["NYC"] - 1, position.Z)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            else if (position.Z >= (int)_parameters["NZC"])
                            {
                                Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, position.Y, (int)_parameters["NZC"] - 1)).ToArray();
                                Particle[] particles = new Particle[originalParticles.Length];
                                Array.Copy(originalParticles, particles, originalParticles.Length);
                                foreach (Particle particle in particles)
                                {
                                    particle.Position.Z = 2.0 * (double)_parameters["ZCV"] - particle.Position.Z;
                                }
                                ghostParticles.AddRange(particles);
                            }
                            
                            
                        }
                    }
                }
            }
            
            if (_domain.Dimensions == 2)
            {
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
                        Vector<int> position = new Vector<int>(cellPosition.X + x, cellPosition.Y + y, 0);

                        if ((position.X < 0) && (position.Y < 0))
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, 0, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = -particle.Position.X;
                                particle.Position.Y = -particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if ((position.X < 0) && (position.Y >= (int)_parameters["NYC"]))
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, (int)_parameters["NYC"] - 1, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = -particle.Position.X;
                                particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if ((position.X >= (int)_parameters["NXC"]) && (position.Y < 0))
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, 0, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                particle.Position.Y = -particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if ((position.X >= (int)_parameters["NXC"]) && (position.Y >= (int)_parameters["NYC"]))
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, (int)_parameters["NYC"] - 1, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                                particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if (position.X < 0)
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(0, position.Y, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = -particle.Position.X;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if (position.Y < 0)
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, 0, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.Y = -particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if (position.X >= (int)_parameters["NXC"])
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>((int)_parameters["NXC"] - 1, position.Y, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.X = 2.0 * (double)_parameters["XCV"] - particle.Position.X;
                            }
                            ghostParticles.AddRange(particles);
                        }
                        else if (position.Y >= (int)_parameters["NYC"])
                        {
                            Particle[] originalParticles = GetListOfParticlesAdjacentToCellPosition(new Vector<int>(position.X, (int)_parameters["NYC"] - 1, 0)).ToArray();
                            Particle[] particles = new Particle[originalParticles.Length];
                            Array.Copy(originalParticles, particles, originalParticles.Length);
                            foreach (Particle particle in particles)
                            {
                                particle.Position.Y = 2.0 * (double)_parameters["YCV"] - particle.Position.Y;
                            }
                            ghostParticles.AddRange(particles);
                        }

                        
                    }
                }
            }

            return ghostParticles;
        }

        public List<Particle> GetListOfGhostParticlesNeighbouringToPosition(Vector<double> position)
        {
            Vector<int> cellPosition = GetCellPositionFromPosition(position);
            return GetListOfGhostParticlesNeighbouringToCellPosition(cellPosition);
        }
    }
}
