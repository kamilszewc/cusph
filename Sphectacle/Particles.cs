using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    /// <summary>
    /// Reprezents a set of particles.
    /// </summary>
    public class Particles
    {
        private Dictionary<string, object> _fields;
        private int _dimensions;

        public int Dimensions { get { return _dimensions; } }

        /// <summary>
        /// Initializes a new instance od Sphectacle.Particles class.
        /// </summary>
        /// <param name="fields">Dictionary of field values organized by name.</param>
        public Particles(Dictionary<string, object> fields)
        {
            _fields = fields;
            if (_fields.ContainsKey("x-position") && _fields.ContainsKey("y-position"))
            {
                if (_fields.ContainsKey("z-position")) _dimensions = 3;
                else _dimensions = 2;
            }
            else
            {
                throw new NoBasicParticlesFieldsException("No basic particle fields (positions)");
            }
        }

        /// <summary>
        /// Returns a single particle (Sphectacle.Particle).
        /// </summary>
        /// <param name="index">Index of particle.</param>
        /// <returns>Instance of particle.</returns>
        public Particle GetParticle(int index)
        {
            Dictionary<string, object> fields = new Dictionary<string, object>();

            foreach (var item in _fields)
            {
                dynamic value = item.Value;
                fields.Add(item.Key, value[index]);
            }

            return new Particle(fields);
        }

        /// <summary>
        /// Returns a single particle (sphectacle.Particle).
        /// </summary>
        /// <param name="index">Index of particle.</param>
        /// <returns>Instance of particle.</returns>
        public Particle this[int index]
        {
            get { return GetParticle(index); }
        }

        /// <summary>
        /// Returns field values of given name.
        /// </summary>
        /// <param name="name">Field name.</param>
        /// <returns>Field values.</returns>
        public object GetField(string name)
        {
            return _fields[name];
        }
        
        
    }
}
