using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class Parameters
    {
        private Dictionary<string, object> _dictionary;
        private int _dimensions;
        private double _h;
        private double _i_h;
        private int _n;

        public int Dimensions { get { return _dimensions; } }
        public double H { get { return _h; } }
        public double I_H { get { return _i_h; } }
        public int N { get { return _n; } }

        public Parameters(Dictionary<string, object> dictionary)
        {
            _dictionary = dictionary;
            if (_dictionary.ContainsKey("NZC")) _dimensions = 3;
            else _dimensions = 2;
            _h = 0.5 * (double)this["XCV"] / (int)this["NXC"];
            _i_h = 1.0 / _h;
            _n = (int)this["N"];
        }

        public object this[string name]
        {
            get { return _dictionary[name]; }
        }
    }
}
