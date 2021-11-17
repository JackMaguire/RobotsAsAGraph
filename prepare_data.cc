#include "deserialize.hh"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;


constexpr auto path_to_all_data = "data/all_data/";

//https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
bool ends_with(
  std::string const & value,
  std::string const & ending
){
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int main(){
  
  std::stringstream training_data;
  std::stringstream validation_data;

  int counter = 0;
  for( auto const & entry : fs::directory_iterator( path_to_all_data ) ){
    if( ends_with( entry.path(), "gz" ) ) continue;
    
    //std::cout << entry.path() << std::endl;

    std::stringstream & stream = (
      counter == 0 ?
      validation_data :
      training_data
    );
    counter = (counter+1)%5;

    std::ifstream input( entry.path() ); //RAII close
    for( std::string line; getline( input, line ); ) {
      DataPoint const sample = deserialize( line );
      if( sample.key == Key::NONE || sample.key == Key::DELETE || sample.key == Key::R ) continue;


      stream << line << '\n';
    }
  }

  std::cout << training_data.str().size() << " " << validation_data.str().size() << std::endl;
}
