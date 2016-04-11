using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sphectacle
{
    public class Vector<T>
    {
        public T X { set; get; }
        public T Y { set; get; }
        public T Z { set; get; }

        public Vector(T x, T y, T z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public double Length
        {
            get
            {
                return System.Math.Sqrt(Math.Pow(Convert.ToDouble(X), 2) + Math.Pow(Convert.ToDouble(Y), 2) + Math.Pow(Convert.ToDouble(Z), 2));
            }
        }

        public static Vector<T> operator +(Vector<T> a, Vector<T> b)
        {
            return new Vector<T>((dynamic)a.X + (dynamic)b.X, (dynamic)a.Y + (dynamic)b.Y, (dynamic)a.Z + b.Z);
        }

        public static Vector<T> operator -(Vector<T> a, Vector<T> b)
        {
            return new Vector<T>((dynamic)a.X - (dynamic)b.X, (dynamic)a.Y - (dynamic)b.Y, (dynamic)a.Z - (dynamic)b.Z);
        }

        public static double operator *(Vector<T> a, Vector<T> b)
        {
            return (dynamic)a.X * (dynamic)b.X + (dynamic)a.Y * (dynamic)b.Y + (dynamic)a.Z * (dynamic)b.Z;
        }

        public static Vector<double> operator %(Vector<T> a, Vector<T>b)
        {
            return new Vector<double>((dynamic)a.Y * (dynamic)b.Z - (dynamic)a.Z * (dynamic)b.Y, 
                                      (dynamic)a.Z * (dynamic)b.X - (dynamic)a.X * (dynamic)b.Z, 
                                      (dynamic)a.X * (dynamic)b.Y - (dynamic)a.Y * (dynamic)b.X);
        }

        public override string ToString()
        {
            return "x=" + X.ToString() + " y=" + Y.ToString() + " z=" + Z.ToString();
        }
    }
}
