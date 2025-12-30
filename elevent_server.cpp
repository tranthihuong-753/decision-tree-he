// Sử dụng rotate_vector để xoay vector trong CKKS với điều kiện {60, 40xn, 60}

#include "examples.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <set>
#include <algorithm> 
#include <chrono>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
using namespace std;
using namespace seal;
using json = nlohmann::json;

// change 
size_t number_threshold = 31; // Số ngưỡng để thử nghiệm trong train_decision_tree() thực tế số ngưỡng N x k = 70 x 4 = 280 
size_t max_depth = 2;
size_t ALL_SAMPLES = 111; // tổng số mẫu trong dữ liệu
size_t NUM_SAMPLES_TRAIN = static_cast<size_t>(ALL_SAMPLES*0.8); 
size_t NUM_SAMPLES_TEST = ALL_SAMPLES - NUM_SAMPLES_TRAIN;
size_t num_label = 2; // su dung cho ham predict_decision_tree()
string src_data = "C:/hu/decision-tree-he/Build_Project/Release/cats_vs_dogs_dataset_resnet_mini_100.csv";
string src_model_tree = "C:/hu/decision-tree-he/model_tree/model_tree_eight_1.bin"; // C:\hu\decision-tree-he\Build_Project\model_tree_sevent.bin
map<string, int> label_map = {
    {"0", 0}, // "cat" -> 0
    {"1", 1} // "dog" -> 1
};

// not change 
size_t poly_modulus_degree = pow(2, 15); 
vector<int> coeff_modulus_bits = 
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40} ; // 18x40  
double scale = pow(2.0, 40);
const vector<double> SOFT_STEP_COEFFICIENTS_16 = {
    5.00000000e-01,
    2.11445799e+00,
    1.15591931e-10,
    -6.38009501e+00,
    -4.91650318e-10,
    1.09534390e+01,
    8.95471329e-10,
    -1.01295272e+01,
    -8.38669100e-10,
    5.27558906e+00,
    4.36355800e-10,
    -1.54908047e+00,
    -1.27390646e-10,
    2.39094701e-01,
    1.95169389e-11,
    -1.50730122e-02,
    -1.22081599e-12
};
const vector<double> SOFT_STEP_COEFFICIENTS_8 = {
    0.5, 
    1.23986659, 
    -0.0, 
    -1.05984904, 
    -0.0, 
    0.40068769, 
    0.0, 
    -0.05021892, 
    0.0
};

// Cau truc de doc tung mau (example) gom K feature, 1 label 
struct Sample {
    vector<double> features; 
    string label;           
};

// Doc du lieu trong file.csv 
// Input: ten file string filename 
// Output: (N_TRAIN+N_TEST) Sample vector<Sample> data 
vector<Sample> read_csv_dynamic(string &filename){
    cout << "read_csv_dynamic()" << endl; 
    vector<Sample> data;
    ifstream file(filename);
    if(!file.is_open()){
        throw runtime_error("\tKhong mo duoc file: " + filename);
    }
    string line;
    while(getline(file, line)){
        stringstream ss(line);
        string item;
        Sample sample;
        vector<string> tokens;
        while(getline(ss, item, ',')){
            tokens.push_back(item);
        }
        if(tokens.empty()) continue;
        cout << "\tLay tat ca cot tru cot cuoi lam features" << endl;
        for(size_t i = 0 ; i < tokens.size() - 1 ; ++i){
            sample.features.push_back(stod(tokens[i]));
        }
        cout << "\tLay cot cuoi lam lable" << endl;
        sample.label = tokens.back();
        data.push_back(sample);
    }
    cout << "compreted read_csv_dynamic()" << filename << endl;
    file.close();
    return data;
}

// Chuan hoa ma tran feature ve khoang [-1, 1] theo cot 
// Input: (N_TRAIN+N_TEST) Sample vector<Sample> data 
// Output: (N_TRAIN+N_TEST) Sample vector<Sample> data co features da chuan hoa ve khoang gia tri [-1, 1]
vector<Sample> normalize(vector<Sample> &data){
    int num_features = data[0].features.size();
    for(int i = 0 ; i < num_features ; i++){
        double min_val = 1e9, max_val = -1e9;
        //Tinh min, max cho tung cot 
        for(auto &row : data){
            min_val = min(min_val, row.features[i]);
            max_val = max(max_val, row.features[i]);
        }
        if (abs(max_val - min_val) < 1e-12) continue; // tranh chia cho 0 
        //Chuan hoa ve khoang [-1, 1]
        for(auto &row : data){
            row.features[i] = 2*(row.features[i]-min_val)/(max_val-min_val)-1;
        }
    }
    return data;
}

struct SplitData{
    vector<Sample> train, test;
};

// Chia du lieu thanh 2 tap train, test theo ty le train_ratio
// Input: vector<Sample>& data
// Output: SplitData data // ban chat la vector<Sample> train, test
SplitData split_data(vector<Sample>& data, double train_ratio = 0.8){
    SplitData split;
    size_t n = data.size();
    size_t n_train = static_cast<size_t>(train_ratio * n);

    cout << "Chia train" << endl;
    for(size_t i=0 ; i<n_train ; i++)
        split.train.push_back(data[i]);

    cout << "Chia test" << endl;
    for (size_t i = n_train; i < n; ++i)
        split.test.push_back(data[i]);
    return split;
}

