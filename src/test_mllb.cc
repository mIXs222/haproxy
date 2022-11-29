// g++ -std=c++20 -Iinclude src/test_mllb.cc -o test_mllb
// ./test_mllb examples/params_test.txt

#include <iostream>
#include <Eigen/Dense>
#include "haproxy/mllb.h"
#include "src/mllb.cc"
 
using Eigen::MatrixXd;
using Eigen::VectorXd;
 
int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Invalid argument length" << std::endl;
    exit(1);
  }
  std::string param_path(argv[1]);
  std::cout << "Parsing " << param_path << std::endl;
  auto param_dict = mllb::parse_param_dict(param_path);
  for (const auto& param : param_dict) {
    std::cout << param.first << ": " << param.second << std::endl;
  }
  mllb::LBNetV1 model(param_dict);
  // MatrixXd server_features = MatrixXd::Random(10, 4);
  MatrixXd server_features(5, 4); server_features <<
    3.0, 3.0, 3.0, 7.0,
    6.0, 5.0, 4.0, 3.0,
    6.0, 5.0, 4.0, 3.0,
    6.0, 5.0, 4.0, 3.0,
    20.0, 10.0, 7.0, 3.0;
  std::cout << "server_features:\n" << server_features << std::endl;
  std::cout << "probabilities: " << model.forward(server_features).transpose() << std::endl;
  std::cout << "action: " << model.select_server(server_features) << std::endl;
}