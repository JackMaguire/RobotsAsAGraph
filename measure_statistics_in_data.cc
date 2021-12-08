//g++ measure_statistics_in_data.cc -o measure_statistics_in_data --std=c++17 -I extern/RobotsCore/include/ -O3 -Wall -pedantic -Wextra -Wshadow

#include "deserialize.hh"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <robots_core/forecasting.hh>

constexpr auto path_to_all_data = "data/all_data/";
constexpr auto outpath_to_training_data = "data/training_data.txt";
constexpr auto outpath_to_validation_data = "data/validation_data.txt";

//https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
bool ends_with(
  std::string const & value,
  std::string const & ending
){
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool move_is_legal(
  Key const move,
  std::array< std::array< robots_core::ForecastResults, 3 >, 3 > const & forecasts
){
  switch( move ){
  case Key::Q: return forecasts[0][2].legal;
  case Key::W: return forecasts[1][2].legal;
  case Key::E: return forecasts[2][2].legal;

  case Key::A: return forecasts[0][1].legal;
  case Key::S: return forecasts[1][1].legal;
  case Key::D: return forecasts[2][1].legal;

  case Key::Z: return forecasts[0][0].legal;
  case Key::X: return forecasts[1][0].legal;
  case Key::C: return forecasts[2][0].legal;

  default: return true;
  }
}

int main(){
  
  std::stringstream training_data;
  std::stringstream validation_data;

  int n_early_teleports = 0;

  std::array< long long unsigned int, 10 > safe_move_hist;
  safe_move_hist.fill( 0 );

  std::array< std::array< long long unsigned int, 10 >, 3 > safe_move_hist_by_level;
  for( auto & a : safe_move_hist_by_level ) a.fill( 0 );
  auto bin_for_level =
    []( int const level ){
      if( level <= 22 ) return 0;
      else if( level <= 44 ) return 1;
      else return 2;
    };


  int counter = 0;
  for( auto const & entry : fs::directory_iterator( path_to_all_data ) ){
    if( ends_with( entry.path(), "gz" ) ) continue;
    
    //std::cout << entry.path() << std::endl;

    bool const validation = counter == 0;
    counter = (counter+1)%5;

    std::stringstream & stream = (
      validation ?
      validation_data :
      training_data
    );

    std::ifstream input( entry.path() ); //RAII close
    for( std::string line; getline( input, line ); ) {
      DataPoint const sample = deserialize( line );

      //Skip move-less keys
      if( sample.key == Key::NONE || sample.key == Key::DELETE || sample.key == Key::R || sample.key == Key::SPACE ) continue;

      auto const forecasts = robots_core::forecast_all_moves(sample.game.board());

      //Skip cascade situations
      if( forecasts[1][1].cascade_safe ) continue;

      //Skip illegal moves
      if( not move_is_legal( sample.key, forecasts ) ) continue;

      unsigned int num_safe_moves = 0;
      for( auto const & f1 : forecasts ){
	for( auto const & f : f1 ){
	  if( f.legal ) ++num_safe_moves;
	}
      }
      //std::cout << "num_safe_moves: " << num_safe_moves << std::endl;
      ++safe_move_hist[ num_safe_moves ];
      ++safe_move_hist_by_level
	[ bin_for_level( sample.level ) ]
	[ num_safe_moves ];

      //Skip obvious teleports
      if( num_safe_moves == 0 ) continue;

      if( num_safe_moves > 0 and sample.key == Key::T ) {
	++n_early_teleports;

	//upsample safe teleports
	stream << line << '\n';
	stream << line << '\n';
	stream << line << '\n';
	stream << line << '\n';
      }

      stream << line << '\n';
    }
  }

  for( unsigned int i = 0; i < safe_move_hist.size(); ++i ){
    std::cout << "safe_move_hist " << i << " " << safe_move_hist[i] << std::endl;
  }

  std::cout << "Levels 1-22:" << std::endl;
  for( unsigned int i = 0; i < safe_move_hist_by_level[0].size(); ++i ){
    std::cout << "safe_move_hist " << i << " " << safe_move_hist_by_level[0][i] << std::endl;
  }

  std::cout << "Levels 23-44:" << std::endl;
  for( unsigned int i = 0; i < safe_move_hist_by_level[1].size(); ++i ){
    std::cout << "safe_move_hist " << i << " " << safe_move_hist_by_level[1][i] << std::endl;
  }

  std::cout << "Levels 45-66:" << std::endl;
  for( unsigned int i = 0; i < safe_move_hist_by_level[2].size(); ++i ){
    std::cout << "safe_move_hist " << i << " " << safe_move_hist_by_level[2][i] << std::endl;
  }

  std::cout << "n_early_teleports: " << n_early_teleports << std::endl;

  std::cout << training_data.str().size() << " " << validation_data.str().size() << std::endl;
}
