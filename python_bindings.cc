//g++ python_bindings.cc -o robots_core$(python3-config --extension-suffix) -O3 -Wall -Wextra -Iinclude -Iextern/RobotsCore/extern/pybind11/include -std=c++17 -fPIC $(python3 -m pybind11 --includes) -shared -Iextern/RobotsCore/include/

#include "deserialize.hh"

#define RC_EXPAND_PYMODULE
#include "core_python_bindings.hh"

//PYBIND11_MODULE(robots_train, m) {
//m.doc() = "GNN Training of the Robots game";
    
py::module m_train = m.def_submodule( "train" );
m_train.def( "make_tensors", &make_tensors );

py::class_< Tensors > tensors( m_train, "Tensors" );
tensors.def_readonly( "input_tensors", &Tensors::input_tensors );
tensors.def_readonly( "output_tensor", &Tensors::output_tensor );


}