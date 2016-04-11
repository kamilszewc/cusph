using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class Domain
    {
        private Particles _particles;
        private Parameters _parameters;
        private string p;

        public Particles Particles { get { return _particles; } }
        public Parameters Parameters { get { return _parameters; } }

        public int Dimensions { get { return _particles.Dimensions; } }

        public int NumberOfCells 
        { 
            get 
            {
                if (_parameters.Dimensions == 3)
                {
                    return (int)_parameters["NXC"] * (int)_parameters["NYC"] * (int)_parameters["NZC"];
                }
                else
                {
                    return (int)_parameters["NXC"] * (int)_parameters["NYC"];
                }
            } 
        }

        
        /// <summary>
        /// Initializes a new instance of Sphectacle.Domain from Particles and Parameters
        /// </summary>
        /// <param name="particles">Particles</param>
        /// <param name="parameters">Parameters</param>
        public Domain(Particles particles, Parameters parameters)
        {
            _particles = particles;
            _parameters = parameters;
        }


        /// <summary>
        /// Initializes a new instance of Spjectacle.Domain from data file
        /// </summary>
        /// <param name="filename">File name</param>
        public Domain(string filename)
        {
            string suffix = filename.Trim().Split('.').LastOrDefault();
            if (suffix == "sph")
            {
                _parameters = DataReader.GetParametersFromSph(filename);
                _particles = DataReader.GetParticlesFromSph(filename);
            }
            else if (suffix == "xml")
            {
                _parameters = DataReader.GetParametersFromXml(filename);
                _particles = DataReader.GetParticlesFromXml(filename);
            }
            else
            {
                throw new WrongFileFormatException("File format is not supported");
            }

        }
        
    }
}
