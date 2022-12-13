#include "haproxy/mllb.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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

std::pair<std::string, ParamDict> parse_param_dict(const std::string &param_path) {
  std::ifstream param_f(param_path);
  ParamDict param_dict;
  std::string model_name; param_f >> model_name;
  // std::cout << "\tmodel_name= " << model_name << std::endl; 
  while(auto named_param = parse_one(param_f)) {
    // std::cout << "Parsed " << named_param->first << ", " << named_param->second << std::endl; 
    param_dict[named_param->first] = named_param->second;
  }
  return std::make_pair(model_name, param_dict);
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

class LBNet {
  // Per-server FCNN with one hidden layer
  Eigen::MatrixXd fc1;  // feature x hidden
  Eigen::MatrixXd policy_fc_last;  // hidden x 1

  // Rng
  std::mt19937 gen;
  std::uniform_real_distribution<double> unif_dist;

public:
  LBNet(const ParamDict &param_dict)
    : fc1(param_dict.at("fc1")),
      policy_fc_last(param_dict.at("policy_fc_last")),
      gen(std::time(nullptr)),
      unif_dist(0.0, 1.0) {
    // std::cout << "LBNet: fc1= " << fc1 << ", policy_fc_last= " << policy_fc_last << std::endl;
  }

  // feature vector --> probability vector
  Eigen::VectorXd forward(Eigen::MatrixXd server_features) {
    // std::cout << "LBNet: server_features.T=\n" << server_features.transpose() << std::endl;
    // std::cout << "LBNet: server_features * fc1= " << server_features * fc1 << std::endl;
    // std::cout << "LBNet: relu(server_features * fc1)= " << relu(server_features * fc1) << std::endl;
    Eigen::MatrixXd logits = relu(server_features * fc1) * policy_fc_last;
    // std::cout << "LBNet: logits= " << logits << std::endl;
    // std::cout << "LBNet: vectorize(logits)= " << vectorize(logits) << std::endl;
    Eigen::VectorXd ps = softmax(vectorize(logits));
    // std::cout << "LBNet: ps.T=\n" << ps.transpose() << std::endl;
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

  virtual int version() = 0;
};

class LBNetV1: public LBNet {
public:
  LBNetV1(const ParamDict &param_dict) : LBNet(param_dict) {}
  int version() override {
    return 1;
  }
};

class LBNetV2: public LBNet {
public:
  LBNetV2(const ParamDict &param_dict) : LBNet(param_dict) {}
  int version() override {
    return 2;
  }
};

LBNet* build_model(std::pair<std::string, ParamDict> args) {
  if (args.first == "lbnetv1") {
    return new LBNetV1(args.second);
  } else if (args.first == "lbnetv2") {
    // lbnetv2 also shares same architecture but has a different input
    return new LBNetV2(args.second);
  }
  std::cerr << "Invalid model_name: " << args.first << std::endl;
  exit(-1);
}

const int HISTORY_LENGTH_V1 = 4;  // TODO: configurable
const int HISTORY_LENGTH_V2 = 10;  // TODO: configurable

class ServerStat {
public:
  void* server_struct_;  // struct server
  int history_length_;
  std::deque<int> num_conns_history_;
  std::deque<double> throughput_history_;
  int current_num_conns_;
  std::chrono::time_point<std::chrono::system_clock> begin_time_;
  int completed_count_;

  ServerStat(void* server_struct, int history_length)
    : server_struct_(server_struct),
      history_length_(history_length),
      num_conns_history_(),
      throughput_history_(),
      current_num_conns_(0),
      begin_time_(std::chrono::system_clock::now()),
      completed_count_(0) {}

  void fill_feature_v1(Eigen::MatrixXd &server_features, int row_idx) {
    assert(server_features.cols() == history_length_);
    assert(num_conns_history_.size() <= history_length_);
    assert(throughput_history_.size() <= history_length_);
    int col_idx = history_length_ - num_conns_history_.size();
    for (int i = std::max(0, (int) num_conns_history_.size() - history_length_); i < num_conns_history_.size(); ++i, ++col_idx) {
      server_features(row_idx, col_idx) = num_conns_history_[i];
    }
  }

  void fill_feature_v2(Eigen::MatrixXd &server_features, int row_idx) {
    assert(server_features.cols() == 2 * history_length_);
    assert(num_conns_history_.size() <= history_length_);
    assert(throughput_history_.size() <= history_length_);
    int col_idx = history_length_ - num_conns_history_.size();
    for (int i = std::max(0, (int) num_conns_history_.size() - history_length_); i < num_conns_history_.size(); ++i, col_idx += 2) {
      server_features(row_idx, col_idx) = num_conns_history_[i];
      server_features(row_idx, col_idx + 1) = throughput_history_[i];
    }
  }

  void take_conn() {
    ++current_num_conns_;
    assert(current_num_conns_ >= 0);
  }

  void drop_conn() {
    --current_num_conns_;
    --current_num_conns_;  // BUG: why see take_conn twice?
    ++completed_count_;
    assert(current_num_conns_ >= 0);
  }

  void record_history() {
    double uptime_ms = (std::chrono::system_clock::now() - begin_time_).count() * 1000;
    num_conns_history_.push_back(current_num_conns_);
    throughput_history_.push_back(completed_count_ / uptime_ms);
    while (num_conns_history_.size() > history_length_) {
      num_conns_history_.pop_front();
      throughput_history_.pop_front();
    }
  }
};

class ModelLB {
  std::unique_ptr<LBNet> model_;
  std::map<int, ServerStat> server_stats_;
  int server_counter_ = 0;
  
  Eigen::MatrixXd get_server_features_v1() {
    Eigen::MatrixXd server_features(server_stats_.size(), HISTORY_LENGTH_V1);
    int row_idx = 0;
    for (auto& [server_id, server_stat] : server_stats_) {
      // std::cout << "fill feature from [" << server_id << "]" << std::endl;
      server_stat.fill_feature_v1(server_features, row_idx++);
    }
    return server_features;
  }
  
  Eigen::MatrixXd get_server_features_v2() {
    Eigen::MatrixXd server_features(server_stats_.size(), 2 * HISTORY_LENGTH_V2);
    int row_idx = 0;
    for (auto& [server_id, server_stat] : server_stats_) {
      // std::cout << "fill feature from [" << server_id << "]" << std::endl;
      server_stat.fill_feature_v2(server_features, row_idx++);
    }
    return server_features;
  }
  
  Eigen::MatrixXd get_server_features() {
    switch (model_->version()) {
      case 1: return get_server_features_v1();
      case 2: return get_server_features_v2();
      default:
        std::cerr << "Invalid model version: " << model_->version() << std::endl;
        exit(-1);
    }
  }

  void* get_server(int idx) {
    assert(idx < server_stats_.size());
    for (auto& [server_id, server_stat] : server_stats_) {
      if (idx-- <= 0) {
        // std::cout << "select_server [" << server_id << "]" << std::endl;
        return server_stat.server_struct_;
      }
    }
    // Unreachable
    return nullptr;
  }

public:
  ModelLB(char param_path[])
    : model_(build_model(parse_param_dict(std::string(param_path)))),
      server_stats_() {}

  ~ModelLB() = default;

  int server_up(void* server_struct) {
    int server_id = ++server_counter_;
    // std::cout << "server_up [" << server_id << "]" << std::endl;
    switch (model_->version()) {
      case 1: server_stats_.emplace(server_id, ServerStat(server_struct, HISTORY_LENGTH_V1));
        break;
      case 2: server_stats_.emplace(server_id, ServerStat(server_struct, HISTORY_LENGTH_V2));
        break;
      default:
        std::cerr << "Invalid model version: " << model_->version() << std::endl;
        exit(-1);
    }
    return server_id;
  }

  void server_down(void* server_struct, int server_id) {
    // std::cout << "server_down [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    auto it = server_stats_.find(server_id);
    assert(it != server_stats_.end());
    assert(it->second.server_struct_ == server_struct);
    server_stats_.erase(it);
  }

  void take_conn(void* server_struct, int server_id) {
    // std::cout << "take_conn [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    server_stats_.find(server_id)->second.take_conn();
  }

  void drop_conn(void* server_struct, int server_id) {
    // std::cout << "drop_conn [" << server_id << "]" << std::endl;
    assert(server_stats_.contains(server_id));
    server_stats_.find(server_id)->second.drop_conn();
  }

  void update_state() {
    for (auto& [server_id, server_stat] : server_stats_) {
      server_stat.record_history();
    }
  }

  void* select_server() {
    return get_server(model_->select_server(get_server_features()));
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