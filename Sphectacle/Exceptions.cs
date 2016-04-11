using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    /// <summary>
    /// The exception that is thrown when no basic particle fields are present
    /// </summary>
    [Serializable()]
    public class NoBasicParticlesFieldsException : System.Exception
    {
        public NoBasicParticlesFieldsException() : base() { }
        public NoBasicParticlesFieldsException(string message) : base(message) { }
        public NoBasicParticlesFieldsException(string message, System.Exception inner) : base(message, inner) { }

        // A constructor is needed for serialization when an
        // exception propagates from a remoting server to the client. 
        protected NoBasicParticlesFieldsException(System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) { }
    }

    /// <summary>
    /// The exception that is thrown when wrong file format is detected
    /// </summary>
    [Serializable()]
    public class WrongFileFormatException : System.Exception
    {
        public WrongFileFormatException() : base() { }
        public WrongFileFormatException(string message) : base(message) { }
        public WrongFileFormatException(string message, System.Exception inner) : base(message, inner) { }

        // A constructor is needed for serialization when an
        // exception propagates from a remoting server to the client. 
        protected WrongFileFormatException(System.Runtime.Serialization.SerializationInfo info,
            System.Runtime.Serialization.StreamingContext context) { }
    }
}
