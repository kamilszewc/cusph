using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Linq;
using System.Globalization;
using System.IO;

namespace Sphectacle
{
    public class DataReader
    {
        private static IEnumerable<string> Split(string fileline)
        {
            string element = "";
            foreach (var ch in (fileline+" "))
            {
                if (ch == ' ')
                {
                    yield return element;
                    element = "";
                }
                else
                {
                    element += ch.ToString();
                }
            }
        }


        private static IEnumerable<XElement> StreamParameters(FileStream fileStream, string elementName)
        {
            using (XmlReader xmlReader = XmlReader.Create(fileStream))
            {
                xmlReader.MoveToContent();

                while (xmlReader.Read())
                {
                    switch (xmlReader.NodeType)
                    {
                        case XmlNodeType.Element:
                            if (xmlReader.Name == elementName)
                            {
                                XElement el = XElement.ReadFrom(xmlReader) as XElement;
                                if (el != null)
                                    yield return el;
                            }
                            break;
                    }
                }
            }
        }


        public static Parameters GetParametersFromSph(string filename)
        {
            Dictionary<string, object> dictionary = new Dictionary<string, object>();

            StreamReader file = new StreamReader(new FileStream(filename, FileMode.Open, FileAccess.Read));

            string fileline;
            bool inParameters = false;
            while ((fileline = file.ReadLine()) != null)
            {
                if (fileline == "") continue;

                string[] dataline = fileline.Trim().Split();
                
                if (dataline[0] == "@parameters")
                {
                    inParameters = true;
                    continue;
                }
                else if ( (dataline[0][0] == '@') && (inParameters == true) )
                {
                    break;
                }

                if (inParameters == true)
                {
                    string name = dataline[0];
                    string type = dataline[1];
                    string value = dataline[2];

                    if (type == "double")
                    {
                        dictionary.Add(name, Convert.ToDouble(value, CultureInfo.InvariantCulture));
                    }
                    else if (type == "int")
                    {
                        dictionary.Add(name, Convert.ToInt32(value, CultureInfo.InvariantCulture));
                    }
                }
            }

            return new Parameters(dictionary);
        }


        public static Parameters GetParametersFromXml(string filename)
        {
            var data = from el in StreamParameters(new FileStream(filename, FileMode.Open, FileAccess.Read), "parameter")
                       select new string[3] { (string)el.Attribute("name"), (string)el.Attribute("type"), (string)el.Value };

            Dictionary<string, object> dictionary = new Dictionary<string,object>();

            foreach (string[] d in data)
            {
                if (d[1] == "double")
                {
                    dictionary.Add(d[0], Convert.ToDouble(d[2], CultureInfo.InvariantCulture));
                }
                else if (d[1] == "int")
                {
                    dictionary.Add(d[0], Convert.ToInt32(d[2], CultureInfo.InvariantCulture));
                }
            }

            return new Parameters(dictionary);
        }


        public static Particles GetParticlesFromSph(string filename)
        {
            StreamReader file = new StreamReader(new FileStream(filename, FileMode.Open, FileAccess.Read));

            Dictionary<string, object> fields = new Dictionary<string, object>();

            string fileline;
            bool inField = false;
            string name = "";
            string type = "";
            while ((fileline = file.ReadLine()) != null)
            {
                if (fileline == "") continue;

                string[] dataline = (from el in Split(fileline.Trim()) select el).ToArray();

                if (dataline[0] == "@field")
                {
                    inField = true;
                    name = dataline[1];
                    type = dataline[2];
                    continue;
                }
                else if ( (dataline[0][0] == '@') && (inField == true) )
                {
                    break;
                }
                
                if (inField == true)
                {
                    if (type == "double")
                    {
                        double[] values = (from el in dataline select Convert.ToDouble(el, CultureInfo.InvariantCulture)).ToArray();
                        fields.Add(name, values);
                    }
                    else if (type == "int")
                    {
                        int[] values = Array.ConvertAll(dataline, Convert.ToInt32);
                        fields.Add(name, values);
                    }
                }
            }

            return new Particles(fields);
        }



        public static Particles GetParticlesFromXml(string filename)
        {
            var data = from el in StreamParameters(new FileStream(filename, FileMode.Open, FileAccess.Read), "field")
                       select new string[3] { (string)el.Attribute("name"), (string)el.Attribute("type"), ((string)el.Value)};

            Dictionary<string, object> fields = new Dictionary<string, object>();
            
            foreach (string[] d in data)
            {
                string name = d[0];
                string type = d[1];
                string[] dataline = d[2].Trim().Split();
                if (type == "double")
                {
                    double[] values = (from el in dataline select Convert.ToDouble(el, CultureInfo.InvariantCulture)).ToArray();
                    fields.Add(name, values);
                }
                else if (d[1] == "int")
                {
                    int[] values = Array.ConvertAll(dataline, Convert.ToInt32);
                    fields.Add(name, values);
                }
            }

            return new Particles(fields);
        }

    }
}
