//g++ python_bindings.cc -o robots_core$(python3-config --extension-suffix) -O3 -Wall -Wextra -Iinclude -Iextern/RobotsCore/extern/pybind11/include -std=c++17 -fPIC $(python3 -m pybind11 --includes) -shared -Iextern/RobotsCore/include/

#include "deserialize.hh"

#define RC_EXPAND_PYMODULE
#include "core_python_bindings.hh"

//PYBIND11_MODULE(robots_train, m) {
//m.doc() = "GNN Training of the Robots game";
    
py::module m_train = m.def_submodule( "train" );
m_train.def( "make_tensors", &make_tensors );
m_train.def( "deserialize", &deserialize );

py::class_< Tensors > tensors( m_train, "Tensors" );
tensors.def_readonly( "input_tensors", &Tensors::input_tensors );
tensors.def_readonly( "output_tensor", &Tensors::output_tensor );

py::enum_< Key >( m_train, "Key" )
      .value( "Q", Key::Q )
      .value( "W", Key::W )
      .value( "E", Key::E )

      .value( "A", Key::A )
      .value( "S", Key::S )
      .value( "D", Key::D )

      .value( "Z", Key::Z )
      .value( "X", Key::X )
      .value( "C", Key::C )

      .value( "T", Key::T )
      .value( "SPACE", Key::SPACE )
      .value( "DELETE", Key::DELETE )

      .value( "NONE", Key::NONE );


py::class_< DataPoint > dp( m_train, "DataPoint" );
dp.def_readonly( "game", &DataPoint::game );
dp.def_readonly( "key", &DataPoint::key );
dp.def_readonly( "level", &DataPoint::level );

}
