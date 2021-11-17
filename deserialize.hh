#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <cassert>

#include <robots_core/game.hh>

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
