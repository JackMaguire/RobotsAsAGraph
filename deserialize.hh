#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <cassert>

#include <robots_core/game.hh>
#include <robots_core/graph/dense.hh>

enum class
Key {
     NONE = 0,
     Q,//1
     W,//2
     E,//3
     A,//4
     S,//5
     D,//6
     Z,//7
     X,//8
     C,//9
     T,
     SPACE,
     DELETE,
     R
};

struct DataPoint {
  robots_core::RobotsGame game;
  Key key = Key::NONE;
};

struct Tensors {
  robots_core::graph::DenseGraph input_tensors;
  std::vector< std::array< float, 1 > > output_tensor;
};

std::vector< std::string > split( std::string const & s, char const delim ){

    std::vector< std::string > result;
    std::stringstream ss( s );
    std::string item;

    while( getline( ss, item, delim ) ){
        result.push_back (item);
    }

    return result;
}

DataPoint
deserialize( std::string const & data_string ){
  DataPoint dp;

  std::vector< std::string > const tokens = split( data_string, ',' );
  //std::cout << data_string << " " << tokens.size() << std::endl;

  if( tokens.size() != 4 ){
    return dp;//NONE
  }

  int const n_tele = std::stoi( tokens[1] );
  int const level = std::stoi( tokens[2] );
  Key const move = Key(std::stoi( tokens[3] ) );

  dp.game.load_from_stringified_representation(
    tokens[0], level, n_tele, 0 );

  dp.key = move;

  //Sanity check:
  //if( move == Key::D ) std::cout << tokens[3] << std::endl;

  return dp;
}

robots_core::graph::SpecialCaseNode
Key2SC( Key const k ){
  using robots_core::graph::SpecialCaseNode;

  switch( k ){
  case( Key::Q ): return SpecialCaseNode::Q;
  case( Key::W ): return SpecialCaseNode::W;
  case( Key::E ): return SpecialCaseNode::E;
  case( Key::A ): return SpecialCaseNode::A;
  case( Key::S ): return SpecialCaseNode::S;
  case( Key::D ): return SpecialCaseNode::D;
  case( Key::Z ): return SpecialCaseNode::Z;
  case( Key::X ): return SpecialCaseNode::X;
  case( Key::C ): return SpecialCaseNode::C;
  case( Key::T ): return SpecialCaseNode::TELEPORT;
  default: assert( false );
  }
  assert( false );
  return SpecialCaseNode::NONE;
}

std::unique_ptr< Tensors >
make_tensors( std::string const & data_string ){
  using namespace robots_core::graph;

  std::unique_ptr< Tensors > t( new Tensors );

  DataPoint const dp = deserialize( data_string );
  t->input_tensors.construct( dp.game );

  SpecialCaseNode const node_to_assign = Key2SC( dp.key );
  t->output_tensor.resize( t->input_tensors.cached_nodes.size() );

  bool match_found = false;

  for( unsigned int i = 0; i < t->output_tensor.size(); ++i ){
    if( t->input_tensors.cached_nodes[i].special_case == node_to_assign ){
      match_found = true;
      t->output_tensor[ i ][0] = 1.0;
    } else {
      t->output_tensor[ i ][0] = 0.0;
    }
  }

  assert( match_found );

  return t;
}
