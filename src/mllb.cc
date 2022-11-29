#include "haproxy/mllb.h"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <string>

#include <Eigen/Dense>

namespace mllb {

typedef std::map<std::string, Eigen::MatrixXd> ParamDict;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenArray;

std::optional<std::pair<std::string, Eigen::MatrixXd>> parse_one(std::ifstream &param_f) {
  std::string name; param_f >> name;
  // std::cout << "\tname= " << name << std::endl; 
  if (name.empty()) return {};
  size_t num_rows, num_cols; param_f >> num_rows >> num_cols;
  // std::cout << "\tnum_rows= " << num_rows << std::endl; 
  // std::cout << "\tnum_cols= " << num_cols << std::endl; 
  Eigen::MatrixXd mat(num_rows, num_cols);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      param_f >> mat(i, j);
    }
  }
  return std::make_pair(name, mat);
}

ParamDict parse_param_dict(const std::string &param_path) {
  std::ifstream param_f(param_path);
  ParamDict param_dict;
  while(auto named_param = parse_one(param_f)) {
    // std::cout << "Parsed " << named_param->first << ", " << named_param->second << std::endl; 
    param_dict[named_param->first] = named_param->second;
  }
  return param_dict;
}

inline Eigen::MatrixXd relu(const Eigen::MatrixXd &x) {
  return x.cwiseMax(0.0);
}

inline Eigen::VectorXd vectorize(const Eigen::MatrixXd &x) {
  assert(x.cols() == 1);
  return x;
}

// https://eigen.tuxfamily.org/bz_attachmentbase/attachment.cgi?id=896
inline Eigen::VectorXd softmax(const Eigen::VectorXd &x) {
  EigenArray xMinusMax = x.rowwise() - x.colwise().maxCoeff();
  return (xMinusMax.rowwise() - xMinusMax.exp().colwise().sum().log()).exp();
}

class LBNetV1 {
  // Per-server FCNN with one hidden layer
  Eigen::MatrixXd fc1;  // feature x hidden
  Eigen::MatrixXd policy_fc_last;  // hidden x 1

  // Rng
  std::mt19937 gen;
  std::uniform_real_distribution<double> unif_dist;

public:
  LBNetV1(const ParamDict &param_dict)
    : fc1(param_dict.at("fc1")),
      policy_fc_last(param_dict.at("policy_fc_last")),
      gen(std::time(nullptr)),
      unif_dist(0.0, 1.0) {
    // std::cout << "LBNetV1: fc1= " << fc1 << ", policy_fc_last= " << policy_fc_last << std::endl;
  }

  // feature vector --> probability vector
  Eigen::VectorXd forward(Eigen::MatrixXd server_features) {
    std::cout << "LBNetV1: server_features.T=\n" << server_features.transpose() << std::endl;
    // std::cout << "LBNetV1: server_features * fc1= " << server_features * fc1 << std::endl;
    // std::cout << "LBNetV1: relu(server_features * fc1)= " << relu(server_features * fc1) << std::endl;
    Eigen::MatrixXd logits = relu(server_features * fc1) * policy_fc_last;
    // std::cout << "LBNetV1: logits= " << logits << std::endl;
    // std::cout << "LBNetV1: vectorize(logits)= " << vectorize(logits) << std::endl;
    Eigen::VectorXd ps = softmax(vectorize(logits));
    std::cout << "LBNetV1: ps.T=\n" << ps.transpose() << std::endl;
    return ps;
  }
  
  // probability vector --> selected integer
  int weighted_random(Eigen::VectorXd probabilities) {
    double r = unif_dist(gen);
    for (int i = 0; i < probabilities.size(); ++i) {
      if (r <= probabilities[i]) {
        return i;
      }
      r -= probabilities[i];
    }
    // Shouldn't reach here
    return probabilities.size() - 1;
  }

  // server_conns: server x feature
  int select_server(Eigen::MatrixXd server_features) {
    return weighted_random(forward(server_features));
  }
};

const int HISTORY_LENGTH = 4;  // TODO: configurable

class ServerStat {
public:
  void* server_struct_;  // struct server
  std::deque<int> num_conns_history_;
  int current_num_conns_;

  ServerStat(void* server_struct) : server_struct_(server_struct), num_conns_history_(), current_num_conns_(0) {}
  void fill_feature(Eigen::MatrixXd &server_features, int row_idx) {
    assert(server_features.cols() == HISTORY_LENGTH);
    int col_idx = HISTORY_LENGTH - num_conns_history_.size();
    for (int num_conn : num_conns_history_) {
      server_features(row_idx, col_idx) = num_conn;
      ++col_idx;
    }
  }

  void take_conn() {
    ++current_num_conns_;
    assert(current_num_conns_ >= 0);
  }

  void drop_conn() {
    --current_num_conns_;
    assert(current_num_conns_ >= 0);
  }

  void record_history() {
    num_conns_history_.push_back(current_num_conns_);
    while (num_conns_history_.size() > HISTORY_LENGTH) {
      num_conns_history_.pop_front();
    }
  }
};

class ModelLB {
  LBNetV1 model_;
  std::map<int, ServerStat> server_stats_;
  int server_counter_ = 0;
  
  Eigen::MatrixXd get_server_features() {
    Eigen::MatrixXd server_features(server_stats_.size(), HISTORY_LENGTH);
    int row_idx = 0;
    for (auto& [server_id, server_stat] : server_stats_) {
      // std::cout << "fill feature from [" << server_id << "]" << std::endl;
      server_stat.fill_feature(server_features, row_idx++);
    }
    return server_features;
  }

  void* get_server(int idx) {
    assert(idx < server_stats_.size());
    for (auto& [server_id, server_stat] : server_stats_) {
      if (idx-- <= 0) {
        std::cout << "select_server [" << server_id << "]" << std::endl;
        return server_stat.server_struct_;
      }
    }
    // Unreachable
    return nullptr;
  }

public:
  ModelLB(char param_path[])
    : model_(LBNetV1(
        parse_param_dict(std::string(param_path))
      )),
      server_stats_() {}

  ~ModelLB() = default;

  int server_up(void* server_struct) {
    int server_id = ++server_counter_;
    std::cout << "server_up [" << server_id << "]" << std::endl;
    server_stats_.emplace(server_id, ServerStat(server_struct));
    return server_id;
  }

  void server_down(void* server_struct, int server_id) {
    std::cout << "server_down [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    auto it = server_stats_.find(server_id);
    assert(it != server_stats_.end());
    assert(it->second.server_struct_ == server_struct);
    server_stats_.erase(it);
  }

  void take_conn(void* server_struct, int server_id) {
    std::cout << "take_conn [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    server_stats_.find(server_id)->second.take_conn();
  }

  void drop_conn(void* server_struct, int server_id) {
    std::cout << "drop_conn [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    server_stats_.find(server_id)->second.drop_conn();
  }

  void update_state() {
    for (auto& [server_id, server_stat] : server_stats_) {
      server_stat.record_history();
    }
  }

  void* select_server() {
    return get_server(model_.select_server(get_server_features()));
  }

};
  
}  // namespace mllb

extern "C" {
void* ModelLB_new(char param_path[]) {
  return new mllb::ModelLB(param_path);
}
void ModelLB_delete(void* mlb_void) {
  delete (mllb::ModelLB*) mlb_void;
}
int ModelLB_server_up(void* mlb_void, void* server_struct) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  return mlb->server_up(server_struct);
}
void ModelLB_server_down(void* mlb_void, void* server_struct, int server_id) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  mlb->server_down(server_struct, server_id);
}
void ModelLB_take_conn(void* mlb_void, void* server_struct, int server_id) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  mlb->take_conn(server_struct, server_id);
}
void ModelLB_drop_conn(void* mlb_void, void* server_struct, int server_id) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  mlb->drop_conn(server_struct, server_id);
}
void ModelLB_update_state(void* mlb_void) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  mlb->update_state();
}
void* ModelLB_select_server(void* mlb_void) {
  mllb::ModelLB *mlb = (mllb::ModelLB*) mlb_void;
  return mlb->select_server();
}
}  // extern "C"