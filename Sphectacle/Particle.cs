using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    
    /// <summary>
    /// Represents a single particle.
    /// </summary>
    public class Particle
    {
        private Vector<double> _position;
        private Dictionary<string, object> _fields;
        private int _dimensions;

        public Vector<double> Position { get { return _position; } }

        public int Dimensions { get { return _dimensions; } }


        /// <summary>
        /// Initializes a new instance of Sphectacle.Particle class.
        /// </summary>
        /// <param name="fields">Dictionary containing the fields.</param>
        public Particle(Dictionary<string, object> fields)
        {
            _fields = fields;
            if ( _fields.ContainsKey("x-position") && _fields.ContainsKey("y-position") )
            {
                if ( _fields.ContainsKey("z-position") )
                {
                    _position = new Vector<double>((double)_fields["x-position"], (double)_fields["y-position"], (double)fields["z-position"]);
                    _dimensions = 3;
                }
                else
                {
                    _position = new Vector<double>((double)_fields["x-position"], (double)_fields["y-position"], 0.0);
                    _dimensions = 2;
                }
            }
            else
            {
                throw new NoBasicParticlesFieldsException("No basic particle fields (positions)");
            }
        }



        /// <summary>
        /// Returns the object of field for given key (field name).
        /// </summary>
        /// <param name="key">Field name</param>
        /// <returns>Field value</returns>
        public object Fields(string key)
        {
            return _fields[key];
        }



        /// <summary>
        /// Returns the object of field for given key (field name).
        /// </summary>
        /// <param name="key">Field name</param>
        /// <returns>Field value</returns>
        public object this[string key]
        {
            get { return Fields(key); }
        }

    }
}
