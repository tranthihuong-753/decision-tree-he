// không tính tổng slot trong cipher 

// acc 100%
// number_threshold = 2; max_depth = 2;
// --- F1 Score Per Class ---
// Class | Precision | Recall | F1-Score
// ------|-----------|--------|-----------
//     0 |    1.0000 | 1.000000 | 1.000000
//     1 |    1.0000 | 1.000000 | 1.000000
//     2 |    1.0000 | 1.000000 | 1.000000
// Macro F1-Score (Average): 1.000000
// Internal Node: Feature Index = 2, Threshold = -0.333333
//   Internal Node: Feature Index = 2, Threshold = -0.333333
//     Leaf Node: [42.211697, 0.152937, 0.007832]
//     Leaf Node: [-1.123178, 0.071805, 0.171323]
//   Internal Node: Feature Index = 3, Threshold = 0.333333
//     Leaf Node: [-1.092619, 35.742440, 7.531804]
//     Leaf Node: [0.004100, 4.032818, 32.289041]

// theta chuan [31 theta]
// voi max_depth = {2, 3, 4, 5} thi acc deu 100% 

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
size_t max_depth = 5;
size_t ALL_SAMPLES = 150; // tổng số mẫu trong dữ liệu
size_t NUM_SAMPLES_TRAIN = static_cast<size_t>(ALL_SAMPLES*0.8); 
size_t NUM_SAMPLES_TEST = ALL_SAMPLES - NUM_SAMPLES_TRAIN;
size_t num_label = 3; // su dung cho ham predict_decision_tree()
string src_data = "C:/hu/decision-tree-he/Build_Project/Release/iris_8_2.csv";

map<string, int> label_map = {
    {"setosa", 0}, // "cat" -> 0
    {"versicolor", 1}, // "dog" -> 1
    {"virginica", 2} // "bird" -> 2
};

// not change 
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

    // cout << "Tao danh sach index 0 den n-1" << endl;
    // vector<size_t> v(n);
    // iota(v.begin(), v.end(), 0);

    // cout << "Random" << endl;
    // random_device rd;
    // mt19937 gen(rd());
    // shuffle(v.begin(), v.end(), gen);

    cout << "Chia train" << endl;
    for(size_t i=0 ; i<n_train ; i++)
        split.train.push_back(data[i]);

    cout << "Chia test" << endl;
    for (size_t i = n_train; i < n; ++i)
        split.test.push_back(data[i]);
    return split;
}

// Chuyen doi du lieu tu Simple sang vecto (T vecto<Simple> sang ma tran) ; chuyen doi label sang onehot 
// Input: vector<Sample>& data
// Output: ma tran du lieu X vector<vector<double>> cua data_X ; cac vector y_one_hot vector<vector<double>> y_one_hot 
pair<vector<vector<double>>, vector<vector<double>>> extract_data(vector<Sample>& data){
    cout << "extract_data()" << endl;
    vector<vector<double>> x_data;
    vector<vector<double>> y_data_one_hot;
    cout << "Chuyen X tu vector<Sample> sang ma tran vector<vector<double>>" << endl;
    cout << "Chuyen y tu double sang one-hot" << endl;
    int count = data.size();
    cout << "So luong mau cua data la " << count << endl;
    for(int i = 0 ; i < count ; i++){
        cout << "Chuyen X[" << i << "] tu Sample sang vector<double> kich thuoc " << data[i].features.size() << ", index=0 co gia tri la " << data[i].features[0] << endl;
        x_data.push_back(data[i].features);
        //file du lieu co label duoc chuyen ve cat=0 , dog=1 
        vector<double> y_onehot(num_label, 0.0);
        if (label_map.count(data[i].label)) {
            int label_index = label_map.at(data[i].label);
            y_onehot[label_index] = 1.0;
        } else {
            cerr << "Canh bao: Nhãn khong hop le: " << data[i].label << endl;
        }
        y_data_one_hot.push_back(y_onehot);
        cout << "Chuyen y[" << i << "] tu double " << data[i].label << " sang one-hot (" << y_onehot[0] << " " << y_onehot[1] << ")" << endl;
    }
    return {x_data, y_data_one_hot};
}

vector<double> soft_step_evaluation(
    const vector<double> &input_values, // z values
    const vector<double> &coefficients // coefficients of the polynomial
)
{
    // 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<vector<double>> powers;
    powers.reserve(coefficients.size()); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(input_values); // z^1
    size_t max_power = coefficients.size() - 1;
    for (size_t k = 1; (1ULL << k) <= max_power; ++k) {
        vector<double> tmp(input_values.size(), 0.0);
        for(size_t i=0 ; i<input_values.size() ; ++i){
            tmp[i] = powers[k-1][i] * powers[k-1][i];
        }
        powers.push_back(tmp);
    }
    // 2. Tính các lũy thừa cần thiết và lưu vào map
    vector<double> result(input_values.size(), 0.0);
    for(size_t i = 0; i < coefficients.size(); ++i) { // i < SOFT_STEP_COEFFICIENTS_16.size()
        double coeff = coefficients[i];
        vector<double> power(input_values.size(), 1.0); // z^0 = 1
        bool first = true;
        for (size_t j = 0; (1ULL << j) <= i; ++j) {
            if (i & (1ULL << j)) {
                if (first) {
                    power = powers[j];
                    first = false;
                } else {
                    for(size_t idx=0 ; idx<input_values.size() ; ++idx){
                        power[idx] *= powers[j][idx];
                    }
                }
            }
        }
        for(size_t idx=0 ; idx<input_values.size() ; ++idx){
            result[idx] += coeff * power[idx];
        }
    }
    return result;
}

struct node{
    bool is_leaf;

    // is_leaf == true
    // L vector<double> leaf_value;
    // Tong slot trong leaf_value la trong so cua nhan l tuong ung 
    vector<vector<double>> leaf_value_vector; 

    // is_leaf == false
    int feature_index; // neu khong phai la node la thi luu index dac trung 
    double threshold; // neu khong phai la node la thi luu nguong 
    unique_ptr<node> left_child;
    unique_ptr<node> right_child;
};

double slot_sum(const vector<double>& vec, int end) {
    double sum = 0.0;
    for (int i = 0; i < end; ++i) {
        sum += vec[i];
    }
    return sum;
}

// Tinh trong so cua tung nhan l trong node la
// Input: vector<double> W_col // vector trong so , vector<vector<double>> Y_cols
// Output: vector<double> weight_sums // vector trong so cua tung nhan l
vector<vector<double>> leaf_value(
    const vector<double> &W_col, // vector trong so 
    const vector<vector<double>> &Y_cols // vector nhan 
){
    // Kiem tra kich thuoc W_col va Y_col phai bang nhau
    int number_W = W_col.size();
    int number_Y = Y_cols[0].size();
    cout << "leaf_value() number_W=" << number_W << ", number_Y=" << Y_cols.size() << " x " << Y_cols[0].size() << endl;    
    if (number_W != number_Y) {
        throw runtime_error("Kich thuoc W_col va Y_col phai bang nhau trong ham leaf_value()");
    }
    vector<vector<double>> output_leaf_values(Y_cols.size(), vector<double>(NUM_SAMPLES_TRAIN, 0.0)); // Kích thước L (số nhãn)
    for(int i=0 ; i< Y_cols.size() ; ++i){
        for(int j=0 ; j< number_W ; ++j){
            output_leaf_values[i][j] = W_col[j] * Y_cols[i][j];
        }
    }
    return output_leaf_values;
}

// Tinh trong so bao mat cho tung nhan l
// OUT : vector<vector<double>> out // size L x N_SAMPLES_TRAIN
// Tong slot trong vector<double> out la trong so bao mat cho tung nhan l tuong ung 
vector<vector<double>> compute_weighted_counts_homo(
    double theta,
    const vector<double> &X_col,
    const vector<double> input_values,
    const vector<double> &coefficients,
    const vector<double> &W_col,          // 1 ciphertextMULTIPLY EXCEPTION
    const vector<vector<double>> &Y_cols   // L ciphertext
){
    vector<vector<double>> out(Y_cols.size(), vector<double>(NUM_SAMPLES_TRAIN, 0.0));
    for(int i=0 ; i< Y_cols.size() ; ++i){
        vector<double> Y_col = Y_cols[i];
        // tinh soft-step 
        vector<double> soft_step_values = soft_step_evaluation(input_values, coefficients);
        // tinh tich w.soft-step.y 
        for(int j=0; j<X_col.size(); ++j){
            out[i][j] = W_col[j] * soft_step_values[j] * Y_col[j];
        }
    }
    return out;
}

double compute_gini_impurity(
    const vector<vector<double>>& right_counts_, // Vector Lx1 cleartext (tổng trọng số bên phải)
    const vector<vector<double>>& left_counts_
)  // Vector Lx1 cleartext (tổng trọng số bên trái)
{
    cout << "compute_gini_impurity()" << endl;
    vector<double> right_counts(right_counts_.size());
    vector<double> left_counts(left_counts_.size());
    for(int l=0 ; l< right_counts.size() ; ++l){
        right_counts[l] = slot_sum(right_counts_[l], NUM_SAMPLES_TRAIN);
        left_counts[l] = slot_sum(left_counts_[l], NUM_SAMPLES_TRAIN);
    }
    // Tính tổng trọng số mẫu ở mỗi bên
    double total_right = 0.0;
    for (double count : right_counts) total_right += count;
    
    double total_left = 0.0;
    for (double count : left_counts) total_left += count;

    double total_all = total_right + total_left;
    if (total_all < 1e-9) return 0.0; // Tránh chia cho 0

    double gini_right = 0.0;
    if (total_right > 1e-9) {
        double sum_sq = 0.0;
        for (double count : right_counts) {
            double prob = count / total_right;
            sum_sq += prob * prob;
        }
        gini_right = (1.0 - sum_sq) * (total_right / total_all);
    }

    double gini_left = 0.0;
    if (total_left > 1e-9) {
        double sum_sq = 0.0;
        for (double count : left_counts) {
            double prob = count / total_left;
            sum_sq += prob * prob;
        }
        gini_left = (1.0 - sum_sq) * (total_left / total_all);
    }

    cout << "Completed compute_gini_impurity()" << endl;
    return gini_right + gini_left;
}

// Tinh W_new_right va W_new_left cho de quy voi vector x 
vector<double> compute_W_phi_best(
    const vector<double>& W_col, 
    const vector<double> input_values,
    const vector<double> &coefficients
)
{
    // tinh soft-step
    vector<double> soft_step_values_right = soft_step_evaluation(input_values, coefficients);
    // tinh W.phi
    vector<double> W_phi(W_col.size(), 0.0);
    for(int i=0 ; i<W_col.size() ; ++i){
        W_phi[i] = W_col[i] * soft_step_values_right[i];
    }
    return W_phi;
}

unique_ptr<node> train_decision_tree(
    const vector<vector<double>> &X_cols,
    const vector<double> &W_col, 
    const vector<vector<double>> &Y_cols,
    const vector<double> &T_cols,
    int depth,
    int max_depth,
    int num_feature
){
    cout << "train_decision_tree() at depth=" << depth << endl;
    // Tao node moi
    unique_ptr<node> node_c = make_unique<node>();
    // Kiem tra dieu kien dung
    if(depth >= max_depth){
        cout << "\tReached max depth. Computing leaf values." << endl;
        node_c->is_leaf = true;
        vector<vector<double>> leaf_values = leaf_value(W_col, Y_cols);
        node_c->leaf_value_vector = leaf_values;
        return node_c;
    }

    // Tim i, theta tot nhat
    double min_gini = 1e9;
    int best_feature = -1;
    double best_threshold = 0.0;

    for(int i=0 ; i<num_feature ; ++i){
        for(int j=0 ; j<number_threshold ; ++j){
            cout << "\tEvaluating feature " << i << " with threshold index " << j << endl;
            double theta = T_cols[j];
            // Tinh input_values_right = X[best_feature] - theta
            vector<double> input_values_right(X_cols[0].size(), 0.0);
            for(int k=0 ; k<X_cols[0].size() ; ++k){
                input_values_right[k] = X_cols[i][k] - theta;
            }
            // Tinh input_values_left = theta - X[best_feature]
            vector<double> input_values_left(X_cols[0].size(), 0.0);
            for(int k=0 ; k<X_cols[0].size() ; ++k){
                input_values_left[k] = theta - X_cols[i][k];
            }
            // Tinh weighted counts cho tung nhan l
            vector<vector<double>> right_counts = compute_weighted_counts_homo(
                theta, X_cols[i], input_values_right, SOFT_STEP_COEFFICIENTS_16, W_col, Y_cols);
            vector<vector<double>> left_counts = compute_weighted_counts_homo(
                theta, X_cols[i], input_values_left, SOFT_STEP_COEFFICIENTS_16, W_col, Y_cols);

            double gini = compute_gini_impurity(right_counts, left_counts);
            // in ra gini 
            cout << "\t\tGini: " << gini << endl;
            if(gini < min_gini){
                // in ra gini 
                cout << "\t\tGini best: " << gini << endl;
                min_gini = gini;
                best_feature = i;
                best_threshold = theta;
            }
        }
    }
    // in ra best_feature va best_threshold
    cout << "\tBest feature: " << best_feature << ", Best threshold: " << best_threshold << endl;

    node_c->feature_index = best_feature;
    node_c->threshold = best_threshold;

    //Tinh input_values = X[best_feature] - theta
    vector<double> input_values_right(X_cols[0].size(), 0.0);
    for(int i=0 ; i<X_cols[0].size() ; ++i){
        input_values_right[i] = X_cols[best_feature][i] - best_threshold;
    }
    // Tinh input_values_left = theta - X[best_feature]
    vector<double> input_values_left(X_cols[0].size(), 0.0);
    for(int i=0 ; i<X_cols[0].size() ; ++i){
        input_values_left[i] = best_threshold - X_cols[best_feature][i];
    }

    // Tinh W_new_right va W_new_left cho de quy voi vector x 
    vector<double> W_phi_right = compute_W_phi_best(W_col, input_values_right, SOFT_STEP_COEFFICIENTS_16);
    vector<double> W_phi_left = compute_W_phi_best(W_col, input_values_left, SOFT_STEP_COEFFICIENTS_16);

    // De quy
    node_c->right_child = train_decision_tree(
        X_cols,
        W_phi_right,
        Y_cols,
        T_cols,
        depth + 1,
        max_depth,
        num_feature
    );
    
    node_c->left_child = train_decision_tree(
        X_cols,
        W_phi_left,
        Y_cols,
        T_cols,
        depth + 1,
        max_depth,
        num_feature
    );

    return node_c;
}

// Ham du doan cho 1 mau
vector<vector<double>> predict_decision_tree(
    const unique_ptr<node>& node_c,       // Nút hiện tại của cây
    const vector<double>& X_cols, // K ciphertext, moi cipher la so 
    int num_feature
)
{
    cout << "predict_decision_tree()" << endl;

    // NÚT LÁ
    if (node_c->is_leaf) {
        cout << " Reached leaf node. Processing leaf values." << endl;
        vector<vector<double>> Leaf_Output = node_c->leaf_value_vector; 
        return Leaf_Output;
    }

    // KHÔNG PHẢI LÁ — TÍNH SOFT-STEP
    int i_best = node_c->feature_index;

    double Theta = node_c->threshold; // so 

    double X_i = X_cols[i_best];

    cout << "Tinh do lech Z = X[best_feature] - theta" << endl;
    double Z_right = X_i - Theta; 
    double Z_left = Theta - X_i; 

    cout << "Tinh soft-step(Z)" << endl;
    double Phi_Right = soft_step_evaluation({Z_right}, SOFT_STEP_COEFFICIENTS_8)[0]; //?SOFT_STEP_COEFFICIENTS_16
    double Phi_Left = soft_step_evaluation({Z_left}, SOFT_STEP_COEFFICIENTS_8)[0]; //?SOFT_STEP_COEFFICIENTS_16

    // GỌI ĐỆ QUY
    vector<vector<double>> Output_Right = predict_decision_tree(
        node_c->right_child, X_cols, num_feature
    );

    vector<vector<double>> Output_Left = predict_decision_tree(
        node_c->left_child, X_cols, num_feature
    );

    // KET QUA 
    vector<vector<double>> Output_Final(Output_Right.size(), vector<double>(Output_Right[0].size(), 0.0));
    for(int l=0 ; l< Output_Right.size() ; ++l){
        for(int m=0 ; m< Output_Right[0].size() ; ++m){
            Output_Final[l][m] = Phi_Right * Output_Right[l][m] + Phi_Left * Output_Left[l][m];
        }
    }

    return Output_Final;
}

// Tinh do chinh xac tren tap test
double calculate_accuracy(
    const vector<vector<double>>& decoded_predictions, // L×N
    const vector<vector<double>>& Y_test_onehot        // N×L
)
{
    cout << "calculate_accuracy()" << endl;
    size_t NUM_LABELS  = decoded_predictions.size();        // L
    cout << " Number of Labels: " << NUM_LABELS << endl;
    // size_t NUM_SAMPLES = decoded_predictions[0].size();     // N
    // cout << " Number of Samples: " << NUM_SAMPLES << endl;

    int correct = 0;

    for (size_t i = 0; i < NUM_SAMPLES_TEST; ++i)
    {
        cout << "Lay du doan Sample " << i << endl;
        vector<double> pred_scores(NUM_LABELS);
        for (size_t l = 0; l < NUM_LABELS; ++l)
            pred_scores[l] = decoded_predictions[l][i];

        int pred_label = 0;
        double max_pred = pred_scores[0];
        for (size_t l = 1; l < NUM_LABELS; ++l)
        {
            if (pred_scores[l] > max_pred) {
                max_pred = pred_scores[l];
                pred_label = l;
            }
        }

        cout << " Tim nhan that Sample " << i << endl;
        const vector<double>& y_true_vec = Y_test_onehot[i];

        int true_label = 0;
        double max_true = y_true_vec[0];
        for (size_t l = 1; l < NUM_LABELS; ++l)
        {
            if (y_true_vec[l] > max_true) {
                max_true = y_true_vec[l];
                true_label = l;
            }
        }

        cout << " Kiem tra chinh xac Sample " << i << endl;
        if (pred_label == true_label)
            correct++;
    }

    double acc = (double)correct / (double)NUM_SAMPLES_TEST;
    cout << "Accuracy: " << fixed << setprecision(6) << acc << endl;
    return acc;
}

// Hàm tính F1-Score cho từng lớp và tính F1-Macro CHƯA SỬ DỤNG 
double calculate_f1_score(
    const vector<vector<double>>& decoded_predictions, // LxN (điểm số dự đoán)
    const vector<vector<double>>& Y_test_onehot // NxL (nhãn thực tế)
)
{
    cout << "calculate_f1_score() - Macro Average" << endl;
    
    // Y_test_onehot là N x L, decoded_predictions là L x N
    const size_t L = decoded_predictions.size();
    // if (L == 0 || decoded_predictions[0].empty()) return 0.0;
    const size_t N = NUM_SAMPLES_TEST;
    cout << " Number of Classes (L): " << L << ", Number of Samples (N): " << N << endl;

    // 1. Xác định Nhãn Dự đoán và Nhãn Thực tế cho từng mẫu
    vector<int> true_labels(N);
    vector<int> predicted_labels(N);

    for (size_t i = 0; i < N; ++i)
    {
        // Nhãn thực tế (Y_test_onehot[i] là vector NxL)
        size_t true_idx = 0;
        double max_true_val = Y_test_onehot[i][0];
        for (size_t l = 1; l < L; ++l) {
            if (Y_test_onehot[i][l] > max_true_val) {
                max_true_val = Y_test_onehot[i][l];
                true_idx = l;
            }
        }
        true_labels[i] = true_idx;

        // Nhãn dự đoán (decoded_predictions[l][i] là vector L x N)
        size_t pred_idx = 0;
        double max_pred_val = decoded_predictions[0][i];
        for (size_t l = 1; l < L; ++l) {
            if (decoded_predictions[l][i] > max_pred_val) {
                max_pred_val = decoded_predictions[l][i];
                pred_idx = l;
            }
        }
        predicted_labels[i] = pred_idx;
    }

    // 2. Tính toán Ma trận Nhầm lẫn (Confusion Matrix)
    // CM[i][j] = số lần lớp i (thực tế) bị dự đoán là lớp j (dự đoán)
    vector<vector<int>> CM(L, vector<int>(L, 0));
    for (size_t i = 0; i < N; ++i) {
        CM[true_labels[i]][predicted_labels[i]]++;
    }

    // 3. Tính Precision, Recall, và F1 cho TỪNG LỚP
    vector<double> f1_scores(L, 0.0);
    double macro_f1 = 0.0;
    
    cout << "\n--- F1 Score Per Class ---" << endl;
    cout << "Class | Precision | Recall | F1-Score" << endl;
    cout << "------|-----------|--------|-----------" << endl;

    for (size_t l = 0; l < L; ++l) {
        // True Positive: Lớp l dự đoán là lớp l
        double TP = CM[l][l];
        
        // False Positive: Lớp khác dự đoán là lớp l (Tổng cột l trừ TP)
        double FP = 0;
        for (size_t i = 0; i < L; ++i) {
            if (i != l) FP += CM[i][l];
        }

        // False Negative: Lớp l dự đoán là lớp khác (Tổng hàng l trừ TP)
        double FN = 0;
        for (size_t j = 0; j < L; ++j) {
            if (j != l) FN += CM[l][j];
        }

        double precision = (TP + FP == 0) ? 0.0 : TP / (TP + FP);
        double recall    = (TP + FN == 0) ? 0.0 : TP / (TP + FN);
        
        double f1 = 0.0;
        if (precision + recall > 1e-9) {
            f1 = 2 * (precision * recall) / (precision + recall);
        }
        
        f1_scores[l] = f1;
        macro_f1 += f1;

        cout << setw(5) << l << " | "
             << fixed << setprecision(4) << setw(9) << precision << " | "
             << fixed << setprecision(6) << setw(6) << recall << " | "
             << fixed << setprecision(6) << setw(8) << f1 << endl;
    }

    // Macro F1 (Trung bình cộng của F1 từng lớp)
    if (L > 0) {
        macro_f1 /= L;
    }

    cout << "\nMacro F1-Score (Average): " << fixed << setprecision(6) << macro_f1 << endl;
    return macro_f1;
}

void print_tree(const unique_ptr<node>& node_c, int depth = 0) {
    if (node_c == nullptr) return;

    // In thụt lề theo độ sâu
    for (int i = 0; i < depth; ++i) {
        cout << "  ";
    }

    if (node_c->is_leaf) {
        // Tinh tong slot trong leaf_value_vector va in ra 
        vector<double> leaf_sums;
        for (const auto& vec : node_c->leaf_value_vector) {
            double sum = slot_sum(vec, NUM_SAMPLES_TRAIN);
            leaf_sums.push_back(sum);
        }
        cout << "Leaf Node: [";
        for (size_t i = 0; i < leaf_sums.size(); ++i) {
            cout << leaf_sums[i];
            if (i < leaf_sums.size() - 1) {
                cout << ", ";
            }
        }
        cout << "]" << endl;

    } else {
        cout << "Internal Node: Feature Index = " << node_c->feature_index
             << ", Threshold = " << node_c->threshold << endl;

        // In cây con bên trái
        print_tree(node_c->left_child, depth + 1);
        // In cây con bên phải
        print_tree(node_c->right_child, depth + 1);
    }
}

int main(){
    try{
        auto start = std::chrono::high_resolution_clock::now();
        print_example_banner("1. LOAD VA CHUAN HOA DU LIEU.");

        cout << "1.1 LOAD DU LIEU TU " << src_data << endl;
        vector<Sample> data = read_csv_dynamic(src_data);
        int num_feature = data[0].features.size();
        cout << "\t So dac trung num_feature = " << num_feature << endl;

        cout << "1.2 CHUAN HOA VE KHOANG [-1, 1]" << endl;
        data = normalize(data);

        cout << "1.3 CHIA TAP TRAIN, TEST." << endl;
        SplitData split = split_data(data);
        cout << "So mau du lieu train la " << split.train.size() << ", moi du lieu co so dac trung la " << split.train[0].features.size() << endl;
        cout << "So mau du lieu test la " << split.test.size() << ", moi du lieu co so dac trung la " << split.train[0].features.size() << endl;

        cout << "1.4 TAO X_train, Y_train_onehot, X_test, Y_test_onehot" << endl;
        // vector<vector<double>>
        auto[X_train, Y_train_onehot] = extract_data(split.train);
        auto[X_test, Y_test_onehot] = extract_data(split.test);

        cout << "1.5 CHUYEN DOI X_train, Y_train_onehot, X_test, Y_test_onehot SANG DANG COT" << endl;
        vector<vector<double>> X_train_T(num_feature, vector<double>(NUM_SAMPLES_TRAIN, 0.0));
        for (int i = 0; i < num_feature; ++i) {
            for (int j = 0; j < NUM_SAMPLES_TRAIN; ++j) {
                X_train_T[i][j] = X_train[j][i];
            }
        }
        vector<vector<double>> X_test_T(num_feature, vector<double>(NUM_SAMPLES_TEST, 0.0));
        for (int i = 0; i < num_feature; ++i) {
            for (int j = 0; j < NUM_SAMPLES_TEST; ++j) {
                X_test_T[i][j] = X_test[j][i];
            }
        }
        vector<vector<double>> Y_train_onehot_T(num_label, vector<double>(NUM_SAMPLES_TRAIN, 0.0));
        for (int i = 0; i < num_label; ++i) {
            for (int j = 0; j < NUM_SAMPLES_TRAIN; ++j) {
                Y_train_onehot_T[i][j] = Y_train_onehot[j][i];
            }
        }
        vector<vector<double>> Y_test_onehot_T(num_label, vector<double>(NUM_SAMPLES_TEST, 0.0));
        for (int i = 0; i < num_label; ++i) {
            for (int j = 0; j < NUM_SAMPLES_TEST; ++j) {
                Y_test_onehot_T[i][j] = Y_test_onehot[j][i];
            }
        }

        cout << "1.6 TAO VECTOR TRONG SO W_col VA NGUONG T_cols" << endl;
        vector<double> W_col(NUM_SAMPLES_TRAIN, 1.0); // vector
        vector<double> T_cols;
        double min_T = -1.0;
        double max_T = 1.0;
        double step = (max_T - min_T) / (number_threshold + 1);
        for(int i=1 ; i<= number_threshold ; ++i){
            double val = min_T + i*step;
            // if(val >= -0.2 && val <= 0.2) continue; 
            T_cols.push_back(val);
        }

        cout << "2. TRAINING." << endl;
        unique_ptr<node> model_tree = train_decision_tree(
            X_train_T,
            W_col,
            Y_train_onehot_T,
            T_cols,
            0,
            max_depth,
            num_feature
        );
        cout << "3. PREDICTION." << endl;
        vector<vector<double>> decoded_predictions(num_label, vector<double>(NUM_SAMPLES_TEST, 0.0)); // L x N
        for(int i=0 ; i< NUM_SAMPLES_TEST ; ++i){
            cout << " Predict Sample " << i << endl;
            vector<double> X_sample(num_feature, 0.0);
            for(int j=0 ; j< num_feature ; ++j){
                X_sample[j] = X_test_T[j][i];
            }
            vector<vector<double>> pred = predict_decision_tree(
                model_tree,
                X_sample,
                num_feature
            );
            for(int l=0 ; l< num_label ; ++l){
                decoded_predictions[l][i] = slot_sum(pred[l], NUM_SAMPLES_TRAIN); // chi co 1 slot
            }
        }
        cout << "4. EVALUATION." << endl;
        double accuracy = calculate_accuracy(
            decoded_predictions,
            Y_test_onehot
        );
        double f1_macro = calculate_f1_score(
            decoded_predictions,
            Y_test_onehot
        );

        print_tree(model_tree);

    }catch(const exception &e){
        cerr << "Exception: " << e.what() << endl;
    }
}
