#include <iostream>
#include <string>

#include "deserialize.hh"

int main(){
  for( std::string line; getline( std::cin, line ); ){
    //std::cout << line << std::endl;

    auto const t = make_tensors( line );
    std::cout << t->input_tensors.x.size() << " " << t->input_tensors.n_edges << std::endl;
  }
}
