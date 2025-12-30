// Sử dụng dog_cat nhưng việc mã hóa onehot cho từng feature tốn RAM quá , server không tải được 
// Sử dụng rotate_vector để xoay vector trong CKKS với điều kiện {60, 40xn, 60}
// Sử dụng soft-step với GPU 

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
// #include <nlohmann/json.hpp>
#include <filesystem>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
using namespace std;
using namespace seal;
// using json = nlohmann::json;

// change 
size_t number_threshold = 31; // Số ngưỡng để thử nghiệm trong train_decision_tree() thực tế số ngưỡng N x k = 70 x 4 = 280 
size_t max_depth = 2;
size_t ALL_SAMPLES = 111; // tổng số mẫu trong dữ liệu
size_t NUM_SAMPLES_TRAIN = static_cast<size_t>(ALL_SAMPLES*0.8); 
size_t NUM_SAMPLES_TEST = ALL_SAMPLES - NUM_SAMPLES_TRAIN;
size_t num_label = 2; // su dung cho ham predict_decision_tree()
// string src_data = "/home/tthuong/decision-tree-he/Build_Project/cats_vs_dogs_dataset_resnet_mini_100.csv";
// string src_data = "C:/hu/decision-tree-he/Build_Project/Release/cats_vs_dogs_dataset_resnet_mini_100.csv";
string src_data = "C:/hu/decision-tree-he/Build_Project/Release/cats_vs_dogs_dataset_resnet_mini_test_2.csv";
string src_model_tree = "C:/hu/decision-tree-he/model_tree/model_tree_elevent_1.bin"; // C:\hu\decision-tree-he\Build_Project\model_tree_sevent.bin
// map<string, int> label_map = {
//     {"0", 0}, // "cat" -> 0
//     {"1", 1} // "dog" -> 1
// };
map<string, int> label_map = {
    {"0", 0}, // "cat" -> 0
    {"1", 1} // "dog" -> 1
};

// not change 
size_t poly_modulus_degree = pow(2, 15); 
vector<int> coeff_modulus_bits = 
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}; // 11x40  
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

// Ma hoa du lieu X, Y, W cho train 
// Input: ma tran chuan cua X vector<vector<double>>& X, const vector<vector<double>>& Y_one_hot
// Output: cipher duoc ma hoa theo cot cua X vector<Ciphertext> C_X_cols, cipher duoc ma hoa theo cot cua Y vector<Ciphertext> C_Y_cols, Ciphertext C_W_col
tuple<vector<Ciphertext>, vector<Ciphertext>, Ciphertext> encrypt_x_y_w(int num_feature, int num_sample, int num_label, double scale, const vector<vector<double>>& X, const vector<vector<double>>& Y_one_hot, const CKKSEncoder &encoder, Encryptor &encryptor){
    cout << "\tMa hoa vector trong so W_train..." << endl;
    vector<double> W_train(num_sample, 1.0);
    Ciphertext C_W_col;
    Plaintext ptw;
    encoder.encode(W_train, scale, ptw); 
    encryptor.encrypt(ptw, C_W_col);
    cout << "\tMa hoa cac dac trung X_train va Y_train..." << endl;
    vector<vector<double>> X_train_T(num_feature, vector<double>(NUM_SAMPLES_TRAIN, 0.0)); // ma trận KxN
    vector<vector<double>> Y_train_T(num_label, vector<double>(NUM_SAMPLES_TRAIN, 0.0)); // ma trận LxN
    for (size_t i = 0; i < num_sample; ++i) {
        for (size_t j = 0; j < num_feature ; ++j) {
            X_train_T[j][i] = X[i][j]; // Đặc trưng j, mẫu i
        }
        for (size_t l = 0; l < num_label ; ++l) {
            Y_train_T[l][i] = Y_one_hot[i][l]; // Nhãn l, mẫu i
        }
    }
    vector<Ciphertext> C_X_cols(num_feature); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < num_feature ; ++j) {
        cout << "\tMa hoa C_X_cols[" << j << "]" << endl;
        Plaintext ptx;
        encoder.encode(X_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols[j]);
    }
    vector<Ciphertext> C_Y_cols(num_label); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < num_label; ++j) {
        cout << "\tMa hoa C_Y_col[" << j << "]" << endl;
        Plaintext ptx;
        encoder.encode(Y_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_Y_cols[j]);
    }
    return make_tuple(C_X_cols, C_Y_cols, C_W_col);
}

// Ma hoa du lieu X, Y cho test 
// giong ma hoa train nhung theo tung nhan 
vector<vector<Ciphertext>> encrypt_x(int num_feature, int num_label, double scale, const vector<vector<double>>& X, const CKKSEncoder &encoder, Encryptor &encryptor){
    vector<vector<Ciphertext>> x_out(NUM_SAMPLES_TEST, vector<Ciphertext>(num_feature)); // NUM_SAMPLES_TEST x num_feature 
    for(int i=0 ; i< NUM_SAMPLES_TEST ; i++){
        cout << "Ma hoa mau thu " << i << " trong tap test" << endl;
        cout << "\tMa hoa X_test thu " << i << endl;
        for(int j=0 ; j< num_feature ; j++){
            cout << "\t\tMa dac trung thu " << j << endl;
            Plaintext ptx;
            encoder.encode(X[i][j], scale, ptx); 
            Ciphertext ctx;
            encryptor.encrypt(ptx, ctx);
            x_out[i][j] = ctx;
        }
    }
    return x_out;
}

// Hàm lấy thông tin của 1 ciphertext 
void print_ct_info(const SEALContext &context, const Ciphertext &ct, const std::string &name) {
    auto context_data = context.get_context_data(ct.parms_id());
    size_t chain_index = context_data->chain_index();
    cout << name << ": chain_index=" << chain_index 
         << ", scale=" << ct.scale() << ", pol_mod_deg=" << context_data->parms().poly_modulus_degree()
         << endl;
}

// Giải mã Ciphertext CKKS sang vector<double>
vector<double> decrdecrypt_to_vectorypt_to_vector(
    const Ciphertext &ct,
    Decryptor &decryptor,
    const CKKSEncoder &encoder
) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);  // Giải mã sang plaintext
    vector<double> result;
    encoder.decode(pt, result); // Giải mã CKKS sang vector<double>
    return result;
}
// vector<double> values = decrypt_to_vector(C_Phi_Left, decryptor, encoder);
// cout << "Values: ";
// for (double v : values) {
//     cout << v << " ";
// }
// cout << endl;

// ham in ra gia tri ma hoa cua 1 ciphertext 
void show_cipher(const Ciphertext &ct, int number, Decryptor &decryptor, const CKKSEncoder &encoder){
    Plaintext pt;
    decryptor.decrypt(ct, pt);  // Giải mã sang plaintext
    vector<double> result;
    encoder.decode(pt, result);
    for(int i=0 ; i < number ; i++){
        cout << result[i] << " ";
    }
    cout << endl;
}

// ham nhan 2 ciphertext an toan 
Ciphertext multiply_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
    cout << "multiply_ciphertexts()" << endl;
    Ciphertext result;
    Ciphertext ct1_copy = ct1;
    Ciphertext ct2_copy = ct2;
    // check chain_index
    auto context_data1 = context.get_context_data(ct1.parms_id());
    auto context_data2 = context.get_context_data(ct2.parms_id());
    size_t chain_index1 = context_data1->chain_index();
    size_t chain_index2 = context_data2->chain_index();
    if (chain_index1 > chain_index2) {
        evaluator.mod_switch_to_inplace(ct1_copy, ct2.parms_id());
    } else if (chain_index2 > chain_index1) {
        evaluator.mod_switch_to_inplace(ct2_copy, ct1.parms_id());
    }
    // check scale
    double target_scale = min(ct1_copy.scale(), ct2_copy.scale());
    ct1_copy.scale() = target_scale;
    ct2_copy.scale() = target_scale;
    // check parms_id
    auto target_parms = ct1_copy.parms_id();
    if (ct2_copy.parms_id() != target_parms) {
        evaluator.mod_switch_to_inplace(ct2_copy, target_parms);
    }
    evaluator.multiply(ct1_copy, ct2_copy, result);
    evaluator.relinearize_inplace(result, relin_keys);
    evaluator.rescale_to_next_inplace(result);

    cout << "completed multiply_ciphertexts()" << endl;
    return result;
}

// ham nhan cipherxplain an toan
Ciphertext multiply_cipher_plain(
    const Ciphertext &ct,
    const Plaintext &pt,
    Evaluator &evaluator,
    const SEALContext &context
) {
    cout << "multiply_cipher_plain()" << endl;

    Ciphertext result;
    Ciphertext ct_copy = ct;
    Plaintext pt_copy = pt;

    // ====== 1. Lấy thông tin context ======
    auto ct_context_data = context.get_context_data(ct.parms_id());
    auto pt_context_data = context.get_context_data(pt.parms_id());

    size_t ct_chain = ct_context_data->chain_index();
    size_t pt_chain = pt_context_data->chain_index();

    // ====== 2. Cân bằng chain index ======
    if (ct_chain > pt_chain) {
        evaluator.mod_switch_to_inplace(ct_copy, pt.parms_id());
    } else if (pt_chain > ct_chain) {
        evaluator.mod_switch_to_inplace(pt_copy, ct.parms_id());
    }

    // ====== 3. Cân bằng scale ======
    // Plaintext scale KHÔNG tự đổi → ta đặt scale của plaintext = scale của ct_copy
    pt_copy.scale() = ct_copy.scale();

    // ====== 4. Đảm bảo parms_id đồng bộ ======
    if (pt_copy.parms_id() != ct_copy.parms_id()) {
        evaluator.mod_switch_to_inplace(pt_copy, ct_copy.parms_id());
    }

    // ====== 5. Nhân an toàn ======
    evaluator.multiply_plain(ct_copy, pt_copy, result);

    cout << "completed multiply_cipher_plain()" << endl;

    return result;
}

// ham cong an toan 
Ciphertext add_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const SEALContext &context
) {
    cout << "add_ciphertexts()" << endl;
    Ciphertext result;
    Ciphertext ct1_copy = ct1;
    Ciphertext ct2_copy = ct2;
    // check chain_index
    auto context_data1 = context.get_context_data(ct1.parms_id());
    auto context_data2 = context.get_context_data(ct2.parms_id());
    size_t chain_index1 = context_data1->chain_index();
    size_t chain_index2 = context_data2->chain_index();
    if (chain_index1 > chain_index2) {
        evaluator.mod_switch_to_inplace(ct1_copy, ct2.parms_id());
    } else if (chain_index2 > chain_index1) {
        evaluator.mod_switch_to_inplace(ct2_copy, ct1.parms_id());
    }
    // check scale
    double target_scale = min(ct1_copy.scale(), ct2_copy.scale());
    ct1_copy.scale() = target_scale;
    ct2_copy.scale() = target_scale;
    // check parms_id
    auto target_parms = ct1_copy.parms_id();
    if (ct2_copy.parms_id() != target_parms) {
        evaluator.mod_switch_to_inplace(ct2_copy, target_parms);
    }
    evaluator.add(ct1_copy, ct2_copy, result);
    cout << "completed add_ciphertexts()" << endl;
    return result;
}

// ham tru an toan ct1-ct2 
Ciphertext sub_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const SEALContext &context
) {
    cout << "sub_ciphertexts()" << endl;
    Ciphertext result;
    Ciphertext ct1_copy = ct1;
    Ciphertext ct2_copy = ct2;
    // check chain_index
    auto context_data1 = context.get_context_data(ct1.parms_id());
    auto context_data2 = context.get_context_data(ct2.parms_id());
    size_t chain_index1 = context_data1->chain_index();
    size_t chain_index2 = context_data2->chain_index();
    if (chain_index1 > chain_index2) {
        evaluator.mod_switch_to_inplace(ct1_copy, ct2.parms_id());
    } else if (chain_index2 > chain_index1) {
        evaluator.mod_switch_to_inplace(ct2_copy, ct1.parms_id());
    }
    // check scale
    double target_scale = min(ct1_copy.scale(), ct2_copy.scale());
    ct1_copy.scale() = target_scale;
    ct2_copy.scale() = target_scale;
    // check parms_id
    auto target_parms = ct1_copy.parms_id();
    if (ct2_copy.parms_id() != target_parms) {
        evaluator.mod_switch_to_inplace(ct2_copy, target_parms);
    }
    evaluator.sub(ct1_copy, ct2_copy, result);

    cout << "completed sub_ciphertexts()" << endl;
    return result;
}

// ham lay tong so phan tu trong vecto cua ciphertext 
// Input: Ciphertext cua vecto<double> ct, so phan tu trong vecto cua cipher number
// Output: Tong cac phan tu cua vecto la so phan tu dau tien cua vecto out 
// VD: input ct=[1, 2, 3, ...]
// out=[6, ...] 
Ciphertext all_cipher_in_index_zero(const Ciphertext &ct, int number, const GaloisKeys &gal_keys, const Evaluator &evaluator){
    cout << "all_cipher_in_index_zero()" << endl;
    Ciphertext res = ct;
    // show_cipher(ct, number+2, decryptor, encoder);
    for (int step = 1; step < number; step+= 1) {
        cout << "\t" << step << endl;
        Ciphertext tmp;
        evaluator.rotate_vector(ct, step, gal_keys, tmp);
        // show_cipher(tmp, number+2, decryptor, encoder);
        evaluator.add_inplace(res, tmp);
    }
    return res;
}

// ham lay tong so phan tu trong vecto<double>
double sum_slots(const vector<double> &vec, int number){
    double sum = 0.0;
    for(int i = 0 ; i < number ; i++){
        sum += vec[i];
    }
    return sum;
}

// ham xoay vector trong CKKS
Ciphertext rotate_to_slot(Ciphertext encrypted, int target_index, const GaloisKeys &gal_keys, Evaluator &evaluator) {
    Ciphertext result;
    // SEAL rotate_vector(input, steps, keys, output)
    // steps > 0: xoay trái (phần tử tại index 'steps' sẽ về index 0)
    // steps < 0: xoay phải
    evaluator.rotate_vector(encrypted, target_index, gal_keys, result);
    return result;
}

Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    const SEALContext &context,
    const vector<double>& SOFT_STEP_COEFFICIENTS
) 
{
    cout << "soft_step_evaluation()" << endl;
    cout << "Tinh luy thua cua z voi phuong phap (exponentiation by squaring) z^k khi k ={1, 2, 4, 8}" << endl;
    // 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(SOFT_STEP_COEFFICIENTS.size()); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // Tính lũy thừa của z với phương pháp bình phương và nhân (exponentiation by squaring) z^k với k ={1, 2, 4, 8}
    for(size_t i = 1; i < SOFT_STEP_COEFFICIENTS.size(); i*=2) {
        cout << " Computing power z^" << i << endl;
        Ciphertext z_for_mul = powers.back();
        Ciphertext tmp = multiply_ciphertexts(
            powers.back(),
            z_for_mul,
            evaluator,
            relin_keys,
            context
        );
        powers.push_back(tmp);
    }
    
    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS[0], scale, c0_plain);
    Ciphertext result;
    encryptor.encrypt(c0_plain, result); 
    evaluator.mod_switch_to_inplace(result, powers.back().parms_id());

    // 2. Tính tổng đa thức: c0 + c1*[[z^1]] + c2*[[z^2]] + ... + c16*[[z^16]]
    cout << "Tinh tong da thuc: c0 + c1*[[z^1]] + c2*[[z^2]] + ... + c16*[[z^16]]" << endl;
    for(size_t i = 1; i < SOFT_STEP_COEFFICIENTS.size(); ++i) { // i < SOFT_STEP_COEFFICIENTS_16.size()
        // Đảm bảo i không vượt quá số lượng hệ số thực tế (nếu SOFT_STEP_COEFFICIENTS_16 nhỏ hơn 5)
        if (i >= SOFT_STEP_COEFFICIENTS.size()) break;
        double coeff = SOFT_STEP_COEFFICIENTS[i];
        if (std::abs(coeff) < 1e-12) continue;

        // Biểu diễn i dưới dạng tổng các lũy thừa của 2
        Ciphertext power;
        bool first = true;
        for (size_t j = 0; (1ULL << j) <= i; ++j) {
            if (i & (1ULL << j)) {
                if (first) {
                    power = powers[j];
                    first = false;
                } else {
                    Ciphertext temp_power = power;
                    temp_power = multiply_ciphertexts(
                        temp_power,
                        powers[j],
                        evaluator,
                        relin_keys,
                        context
                    );
                    power = temp_power;
                }
            }
        }

        // 2.1 Encode hệ số với scale phù hợp (scale cần giống với power.scale())
        double power_scale = power.scale(); 
        Plaintext coeff_plain;
        encoder.encode(coeff, power_scale, coeff_plain);

        // 2.2 Nếu coeff_plain.parms_id() khác với power.parms_id(), mod-switch coeff_plain
        evaluator.mod_switch_to_inplace(coeff_plain, power.parms_id());

        // 2.3 Nhân: term = coeff * z^i
        Ciphertext term = multiply_cipher_plain(
            power,
            coeff_plain,
            evaluator,
            context
        );

        result = add_ciphertexts(
            result,
            term,
            evaluator,
            context
        );
    }
    cout << "Completed soft_step_evaluation()" << endl;
    return result;
}

struct Node_c {
    bool is_leaf = false;
    // Giá trị nút lá (Nếu là lá, kích thước Lx1) leaf_value[l] = trọng số chỉ xác suất rơi vào nhãn l  
    // Tổng slot trong 1 Ciphertext leaf_value là trọng số tương ứng với từng nhãn l
    vector<Ciphertext> leaf_value_vector;

    // Thông số phân chia (Nếu không là lá)
    Ciphertext feature_index; // one-hot vector chỉ số đặc trưng để phân chia
    Ciphertext threshold; // số

    // Các nhánh (Node con)
    unique_ptr<Node_c> left_child = nullptr;
    unique_ptr<Node_c> right_child = nullptr;
};

vector<Ciphertext> leaf_value(
    const Ciphertext &C_W_col,         
    const vector<Ciphertext> &C_Y_cols,        
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const GaloisKeys &gal_keys,
    const SEALContext &context,
    int num_feature
){
    cout << "leaf_value()" << endl;
    vector<Ciphertext> C_leaf_values(num_label);
    for (size_t l = 0; l < num_label ; ++l) {
        cout << "\tProcessing label " << l << endl;
        Ciphertext P = multiply_ciphertexts(
            C_W_col,
            C_Y_cols[l],
            evaluator,
            relin_keys,
            context
        );
        // Tính tổng các slot trong P
        C_leaf_values[l] = P;
    }
    cout << "Completed leaf_value() = " << endl;
    return C_leaf_values;
}

pair<vector<Ciphertext>, vector<Ciphertext>> compute_weighted_counts_homo(
    Ciphertext best_feature, // [Tinh X_col] 1 số plaintext 
    Ciphertext C_T_col, // [theta] 1 số plaintext 
    const vector<Ciphertext>& C_X_cols, // K ciphertext
    const Ciphertext& C_W_col,          // 1 ciphertextMULTIPLY EXCEPTION
    const vector<Ciphertext>& C_Y_cols,   // L ciphertext
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& gal_keys,
    const SEALContext& context,
    int num_feature
){
    cout << "compute_weighted_counts_homo()" << endl;
    vector<Ciphertext> C_right_counts(num_label);
    vector<Ciphertext> C_left_counts(num_label);
    cout << "Tinh sort-step()" << endl;

    cout << "\tTinh C_X[i]" << endl;
    Ciphertext C_X_i;
    bool first = true;
    for (int k = 0; k < num_feature; ++k) {
        Ciphertext reline_best_feature = rotate_to_slot(best_feature, k, gal_keys, evaluator);
        Ciphertext term = multiply_ciphertexts(
            reline_best_feature,
            C_X_cols[k],
            evaluator,
            relin_keys,
            context
        );

        if (first) {
            C_X_i = term;
            first = false;
        } else {
            evaluator.add_inplace(C_X_i, term);
        }
    }

    cout << "\tTinh (C_X[i] - C_T_col) va (C_T_col - C_X[i])" << endl;
    Ciphertext C_Z_right = sub_ciphertexts(C_X_i, C_T_col, evaluator, context);
    Ciphertext C_Z_left = sub_ciphertexts(C_T_col, C_X_i, evaluator, context); 
    
    cout << "\tTinh sort-step() voi input la (C_X[i] - C_T_col) va (C_T_col - C_X[i])" << endl;
    Ciphertext C_Phi_Right = soft_step_evaluation(
        C_Z_right, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    Ciphertext C_Phi_Left = soft_step_evaluation(
        C_Z_left, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16


    cout << "Tinh C_W_col*sort-step()*C_Y_col" << endl;
    Ciphertext C_W_Phi_Right, C_W_Phi_Left;
    C_W_Phi_Right = multiply_ciphertexts(
        C_W_col,
        C_Phi_Right,
        evaluator,
        relin_keys,
        context
    );
    C_W_Phi_Left = multiply_ciphertexts(
        C_W_col,
        C_Phi_Left,
        evaluator,
        relin_keys,
        context
    );
    // Tính W.Phi.Y và tổng trọng số cho 1 nhãn bị chia bởi i, theta 
    cout << "Compute weighted counts per label" << endl;
    for (int l = 0; l < num_label; ++l) {
        // RIGHT
        Ciphertext C_term_right = multiply_ciphertexts(
            C_W_Phi_Right,
            C_Y_cols[l],
            evaluator,
            relin_keys,
            context
        );
        C_right_counts[l] = C_term_right;
        // LEFT
        Ciphertext C_term_left = multiply_ciphertexts(
            C_W_Phi_Left,
            C_Y_cols[l],
            evaluator,
            relin_keys,
            context
        );
        C_left_counts[l] = C_term_left;
    }
    return {C_right_counts, C_left_counts};
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
        right_counts[l] = sum_slots(right_counts_[l], NUM_SAMPLES_TRAIN);
        left_counts[l] = sum_slots(left_counts_[l], NUM_SAMPLES_TRAIN);
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

pair<Ciphertext, Ciphertext> compute_W_phi_best(
    Ciphertext best_feature, //one-hot 
    Ciphertext best_threshold, // so 
    const vector<Ciphertext>& C_X_cols, 
    const Ciphertext& C_W_col, 
    Evaluator& evaluator, 
    Encryptor& encryptor, 
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys, 
    const SEALContext& context, 
    const GaloisKeys& gal_keys,
    int num_feature
)
{
    cout << "compute_W_phi_best()" << endl;

    cout << "Tinh X[best_feature]" << endl;
    Ciphertext C_X_i;
    bool first = true;
    for (int k = 0; k < num_feature; ++k) {
        Ciphertext reline_best_feature = rotate_to_slot(best_feature, k, gal_keys, evaluator);
        Ciphertext term = multiply_ciphertexts(
            reline_best_feature,
            C_X_cols[k],
            evaluator,
            relin_keys,
            context
        );

        if (first) {
            C_X_i = term;
            first = false;
        } else {
            evaluator.add_inplace(C_X_i, term);
        }
    }
    
    cout << "Tinh do lech Z = X[best_feature] - theta" << endl;
    Ciphertext C_Z_right = sub_ciphertexts(C_X_i, best_threshold, evaluator, context); 
    Ciphertext C_Z_left = sub_ciphertexts(best_threshold, C_X_i, evaluator, context); 

    cout << "Tinh soft-step(Z)" << endl;
    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    
    cout << "Tinh W_new = W * Phi" << endl;
    Ciphertext C_W_new_right, C_W_new_left;
    
    // Right: W * Phi_Right
    // align_and_multiply(C_W_col, C_Phi_Right, C_W_new_right);
    C_W_new_right = multiply_ciphertexts(
        C_W_col,
        C_Phi_Right,
        evaluator,
        relin_keys,
        context
    );
    // align_and_multiply(C_W_col, C_Phi_Left, C_W_new_left);
    C_W_new_left = multiply_ciphertexts(
        C_W_col,
        C_Phi_Left,
        evaluator,
        relin_keys,
        context
    );
    
    cout << "Completed compute_W_phi_best()" << endl;
    return {C_W_new_right, C_W_new_left};
}

unique_ptr<Node_c> train_decision_tree(
    const vector<Ciphertext>& C_X_cols,
    const Ciphertext& C_W_col, 
    const vector<Ciphertext>& C_Y_cols,
    const vector<Ciphertext>& C_T_cols, // Tập hợp các ngưỡng duy nhất
    int depth,
    int max_depth,
    Evaluator& evaluator,
    Encryptor& encryptor,
    Decryptor& decryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& gal_keys,
    const SEALContext& context,
    int num_feature, 
    vector<Ciphertext> C_I_one_hot
){
    cout << "train_decision_tree() depth=" << depth << endl;
    auto node_c = make_unique<Node_c>();
    // 1. Dieu kien dung 
    if(depth >= max_depth){
        cout << "Node lead o do sau depth = " << depth << endl;
        node_c->is_leaf = true; 
        vector<Ciphertext> C_leaf_values = leaf_value(
            C_W_col, C_Y_cols, evaluator, relin_keys, gal_keys, context, num_feature
        );
        node_c->leaf_value_vector = C_leaf_values;
        return node_c;
    }
    // Than ham 
    cout << "Tim nguong, dac trung toi uu cho node co do sau depth = " << depth << endl;

    cout << "Server lap qua tung thuoc tinh, tung nguong ma hoa de tinh trong so bao mat cho right, left." << endl;
    vector<vector<vector<Ciphertext>>> C_right_counts_C_T_cols_I, C_left_counts_C_T_cols_I;  // <vector<vector<vector<ciphertext>>>, vector<vector<vector<ciphertext>>>> cipher 1xn , num_feature x num_theta x l x cipher
    for(int i=0 ; i<num_feature ; ++i){
        cout << "\t[Server]Dac trung thu " << i << endl;
        Ciphertext C_one_hot_feature = C_I_one_hot[i];
        vector<vector<Ciphertext>> C_right_counts_C_T_cols, C_left_counts_C_T_cols;  // <vector<vector<ciphertext>>, vector<vector<ciphertext>>> cipher 1xn , kich thuoc num_theta x l x cipher
        int number_threshold=0;
        for(Ciphertext C_T_col:C_T_cols){
            cout << "\t\tNguong thu " << number_threshold << endl;
            ++number_threshold;
            
            cout << "\t\tTinh tong trong so bao mat cho tung nhan. " << endl;
            auto [C_right_counts, C_left_counts] = compute_weighted_counts_homo(C_one_hot_feature, C_T_col, C_X_cols, C_W_col, C_Y_cols, evaluator, encryptor, encoder, relin_keys, gal_keys, context, num_feature);
            C_right_counts_C_T_cols.push_back(C_right_counts);
            C_left_counts_C_T_cols.push_back(C_left_counts);
        }
        C_right_counts_C_T_cols_I.push_back(C_right_counts_C_T_cols);
        C_left_counts_C_T_cols_I.push_back(C_left_counts_C_T_cols);
    }
    cout << "Client giai ma trong so bao mat, tinh best_feature va best_threshold" << endl;
    double min_gini = 1e9;
    int best_feature = -1; // index feature
    double best_threshold = -1; // index nguong 

    int num_theta = C_T_cols.size();
    for(int i=0 ; i<num_feature ; ++i){
        cout << "\t[Client]Dac trung thu " << i << endl;
        vector<vector<Ciphertext>> C_right_counts_C_T_cols_client = C_right_counts_C_T_cols_I[i];
        vector<vector<Ciphertext>> C_left_counts_C_T_cols_client = C_left_counts_C_T_cols_I[i];
        for(int j=0 ; j<num_theta ; ++j){
            cout << "\t[Client]Nguong thu " << j << endl;
            vector<Ciphertext> C_right_counts_client = C_right_counts_C_T_cols_client[j];
            vector<Ciphertext> C_left_counts_client = C_left_counts_C_T_cols_client[j];
            cout << "\tGiai ma va lay trong so tong cua tung nhan l." << endl;
            vector<vector<double>> right_counts_clear(num_label, vector<double>(NUM_SAMPLES_TRAIN));
            vector<vector<double>> left_counts_clear(num_label, vector<double>(NUM_SAMPLES_TRAIN));
            for (size_t l = 0; l < num_label; ++l) {
                cout << "\t\tNhan thu " << l << endl;
                Plaintext pt_r, pt_l;
                decryptor.decrypt(C_right_counts_client[l], pt_r);
                decryptor.decrypt(C_left_counts_client[l], pt_l);
                
                vector<double> decoded_r, decoded_l;
                encoder.decode(pt_r, decoded_r);
                encoder.decode(pt_l, decoded_l);
                
                // right_counts_clear[l] = decoded_r[0]; 
                // left_counts_clear[l] = decoded_l[0];

                right_counts_clear[l] = decoded_r;
                left_counts_clear[l] = decoded_l;

                // cout << "\t\t\tRight=" << decoded_r[0] << endl;
                // cout << "\t\t\tLeft=" << decoded_l[0] << endl;
            }
            
            cout << "\tTinh Gini Impurity" << endl;
            double current_gini = compute_gini_impurity(right_counts_clear, left_counts_clear);

            // 2.4. Cập nhật ngưỡng tốt nhất
            if (current_gini < min_gini) {
                min_gini = current_gini;
                best_feature = i; // index feature
                best_threshold = j; // index nguong       
            }
        }
    }
    cout << "Index best_feature = " << best_feature << " | Index best_threshold = " << best_threshold << endl;
    // Chuyen int best_feature thành ciphertext one-hot
    cout << "Ma hoa best_feature one-hot, best_threshold" << endl;
    Ciphertext C_best_feature = C_I_one_hot[best_feature];
    // Ma hoa int best_threshold
    Ciphertext C_best_threshold = C_T_cols[best_threshold];

    // Cap nhat node 
    node_c->feature_index = C_best_feature; // one-hot
    node_c->threshold = C_best_threshold; // so 
    
    cout << "Tinh w_new cho de quy." << endl;
    auto [C_W_new_right, C_W_new_left] = compute_W_phi_best(
        C_best_feature, C_best_threshold, C_X_cols, C_W_col, 
        evaluator, encryptor, encoder, relin_keys, context, gal_keys, num_feature
    );

    node_c->right_child = train_decision_tree(
        C_X_cols, C_W_new_right, C_Y_cols, C_T_cols, 
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, gal_keys, context, num_feature, C_I_one_hot
    );
    node_c->left_child = train_decision_tree(
        C_X_cols, C_W_new_left, C_Y_cols, C_T_cols,
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, gal_keys, context, num_feature, C_I_one_hot
    );

    cout << "Completed train_decision_tree() at depth=" << depth << endl;
    return node_c;
}

// co the chuyen qua node_c->leaf_value 
// predict cho 1 mẫu 
vector<Ciphertext> predict_decision_tree(
    const unique_ptr<Node_c>& node_c,       // Nút hiện tại của cây
    const vector<Ciphertext>& C_X_cols, // K ciphertext, moi cipher la so 
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const SEALContext& context,
    const GaloisKeys& galois_keys,
    int num_feature
)
{
    cout << "predict_decision_tree()" << endl;

    // NÚT LÁ
    if (node_c->is_leaf) {
        cout << " Reached leaf node. Processing leaf values." << endl;
        vector<Ciphertext> C_Leaf_Output = node_c->leaf_value_vector; 
        return C_Leaf_Output;
    }

    // KHÔNG PHẢI LÁ — TÍNH SOFT-STEP
    Ciphertext i_best = node_c->feature_index;

    Ciphertext C_Theta = node_c->threshold; // so 

    Ciphertext C_X_i;
    bool first = true;
    for (int k = 0; k < num_feature; ++k) {
        Ciphertext reline_best_feature = rotate_to_slot(i_best, k, galois_keys, evaluator);
        Ciphertext term = multiply_ciphertexts(
            reline_best_feature,
            C_X_cols[k],
            evaluator,
            relin_keys,
            context
        );

        if (first) {
            C_X_i = term;
            first = false;
        } else {
            evaluator.add_inplace(C_X_i, term);
        }
    }

    // print_ct_info(context, C_Theta, "C_Theta");
    // print_ct_info(context, C_X_i, "C_X_i");

    Ciphertext C_Z_right, C_Z_left;
    // evaluator.sub(C_X_i, C_Theta, C_Z_right);
    C_Z_right = sub_ciphertexts(C_X_i, C_Theta, evaluator, context);
    // evaluator.sub(C_Theta, C_X_i, C_Z_left);
    C_Z_left = sub_ciphertexts(C_Theta, C_X_i, evaluator, context);

    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_8);
    Ciphertext C_Phi_Left  = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, context, SOFT_STEP_COEFFICIENTS_8);

    // GỌI ĐỆ QUY
    vector<Ciphertext> C_Output_Right = predict_decision_tree(
        node_c->right_child, C_X_cols, evaluator, encryptor, encoder, 
        relin_keys, context, galois_keys, num_feature
    );

    vector<Ciphertext> C_Output_Left = predict_decision_tree(
        node_c->left_child, C_X_cols, evaluator, encryptor, encoder, 
        relin_keys, context, galois_keys, num_feature
    );

    // TỔNG HỢP KẾT QUẢ
    vector<Ciphertext> C_Out_Phi_Right(num_label);
    vector<Ciphertext> C_Out_Phi_Left(num_label);
    for (size_t l = 0; l < num_label ; ++l) {
        C_Out_Phi_Right[l] = multiply_ciphertexts(
            C_Output_Right[l],
            C_Phi_Right,
            evaluator,
            relin_keys,
            context
        );
        C_Out_Phi_Left[l] = multiply_ciphertexts(
            C_Output_Left[l],
            C_Phi_Left,
            evaluator,
            relin_keys,
            context
        );
    }

    vector<Ciphertext> C_Final_Output;
    for (size_t l = 0; l < num_label ; ++l) {
        Ciphertext C_Sum = add_ciphertexts(
            C_Out_Phi_Right[l],
            C_Out_Phi_Left[l],
            evaluator,
            context
        );
        C_Final_Output.push_back(C_Sum);
    }

    cout << "Completed predict_decision_tree()" << endl;
    return C_Final_Output;
}

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

// bool save_log_json(
//     const std::string &filename,
//     double accuracy,
//     size_t NUM_SAMPLES_TRAIN,
//     size_t NUM_SAMPLES_TEST,
//     const std::string &src_data,
//     size_t number_threshold,
//     size_t max_depth,
//     size_t num_label,
//     const std::string &src_model_tree,
//     double time_encrypt,
//     double time_encrypt_thresholds,
//     double time_training,
//     double time_save_model,
//     double time_load_model,
//     double time_predict_model,
//     long long start_time_ms,
//     long long end_time_ms
// ) {
//     json root;

//     // Nếu file tồn tại → load
//     std::ifstream infile(filename);
//     if (infile.good()) {
//         try {
//             infile >> root;
//         } catch (...) {
//             std::cerr << "[WARNING] JSON file corrupted. Creating new file.\n";
//             root = json::array();
//         }
//     } else {
//         // File không tồn tại → tạo mảng JSON rỗng
//         root = json::array();
//     }
//     infile.close();

//     // Tạo entry mới
//     json entry = {
//         {"accuracy", accuracy},
//         {"NUM_SAMPLES_TRAIN", NUM_SAMPLES_TRAIN},
//         {"NUM_SAMPLES_TEST", NUM_SAMPLES_TEST},
//         {"src_data", src_data},
//         {"number_threshold", number_threshold},
//         {"max_depth", max_depth},
//         {"num_label", num_label},
//         {"src_model_tree", src_model_tree},
//         {"time_encrypt", time_encrypt},
//         {"time_encrypt_thresholds", time_encrypt_thresholds},
//         {"time_training", time_training},
//         {"time_save_model", time_save_model},
//         {"time_load_model", time_load_model},
//         {"time_predict_model", time_predict_model},
//         {"start_time_ms", start_time_ms},
//         {"end_time_ms", end_time_ms}
//     };

//     // Append to JSON array
//     root.push_back(entry);

//     // Lưu lại
//     std::ofstream outfile(filename);
//     if (!outfile.is_open()) {
//         std::cerr << "Cannot open model file to save!\n";
//         return false;
//     }

//     outfile << root.dump(4); // indent=4 cho đẹp
//     outfile.close();

//     std::cout << "Log saved to " << filename << "\n";
//     return true;
// }

// auto start = std::chrono::high_resolution_clock::now();
// auto end = std::chrono::high_resolution_clock::now();
// double ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

// void print_indent(int depth) {
//     for (int i = 0; i < depth; ++i)
//         cout << "  ";
// }

// // Hiển thị cấu trúc cây (chỉ thông tin mã hóa)
// void print_tree_structure(
//     const unique_ptr<Node_c> &node,
//     int depth = 0
// ) {
//     if (!node) {
//         print_indent(depth);
//         cout << "(null)" << endl;
//         return;
//     }
//     print_indent(depth);
//     if (node->is_leaf) {
//         cout << "[Leaf] ";
//         cout << "leaf_value_vector size = "
//              << node->leaf_value_vector.size() << endl;
//         return;
//     }
//     cout << "[Node] ";
//     cout << "feature_index size = "
//          << node->feature_index.size() << endl;
//     print_indent(depth);
//     cout << "├─ Left:" << endl;
//     print_tree_structure(node->left_child, depth + 1);
//     print_indent(depth);
//     cout << "└─ Right:" << endl;
//     print_tree_structure(node->right_child, depth + 1);
// }

// Hiển thị cấu trúc cây (sau giải mã)
void print_tree_decrypted(
    const unique_ptr<Node_c>& node_c,
    const SEALContext& context,
    Decryptor& decryptor,
    const CKKSEncoder& encoder,
    int depth = 0
) {
    if (!node_c) return;

    // In thụt lề theo depth
    for (int i = 0; i < depth; ++i) {
        cout << "  ";
    }

    // ===== LEAF NODE =====
    if (node_c->is_leaf) {
        cout << "Leaf Node: [";
        vector<Ciphertext> leaf_values_ct = node_c->leaf_value_vector;
        for(int i=0 ; i<num_label ; ++i){
            Plaintext pt;
            decryptor.decrypt(leaf_values_ct[i], pt);

            vector<double> vals;
            encoder.decode(pt, vals);

            double x = sum_slots(vals, NUM_SAMPLES_TRAIN);

            cout << x;
            if (i < node_c->leaf_value_vector.size() - 1)
                cout << ", ";
        }

        cout << "]" << endl;
        return;
    }

    // ===== INTERNAL NODE =====

    // Giải mã feature_index (one-hot -> index)
    
    Plaintext pt;
    decryptor.decrypt(node_c->feature_index, pt);

    vector<double> val;
    encoder.decode(pt, val);

    int feature_index=0;
    for(int i=0 ; i<val.size() ; i++){
        if(val[i] - 0.5 > 0){
            feature_index=i;
            break;
        }
    }

    // Giải mã threshold
    Plaintext pt_th;
    decryptor.decrypt(node_c->threshold, pt_th);
    vector<double> th_vals;
    encoder.decode(pt_th, th_vals);
    double threshold = th_vals[0];

    cout << "Internal Node: Feature Index = "
         << feature_index
         << ", Threshold = "
         << threshold << endl;

    // In cây con bên trái
    print_tree_decrypted(
        node_c->left_child,
        context,
        decryptor,
        encoder,
        depth + 1
    );

    // In cây con bên phải
    print_tree_decrypted(
        node_c->right_child,
        context,
        decryptor,
        encoder,
        depth + 1
    );
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


        print_example_banner("2. TAO CONTEXT, KEY MA HOA.");
        EncryptionParameters parms(scheme_type::ckks);
        // bộ 3 tham số poly_modulus_degree và coeff_modulus_bits và scale
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(
            poly_modulus_degree,
            coeff_modulus_bits
        ));
        SEALContext context(parms);
        print_parameters(context);
        cout << endl;

        KeyGenerator keygen(context);
        PublicKey public_key;
        keygen.create_public_key(public_key);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        GaloisKeys gal_keys;
        keygen.create_galois_keys(gal_keys);
        Encryptor encryptor(context, public_key);
        Evaluator evaluator(context);
        Decryptor decryptor(context, keygen.secret_key());
        CKKSEncoder encoder(context);


        print_example_banner("3. MA HOA DU LIEU.");
        size_t slot_count = encoder.slot_count();
        cout << "SO LUONG MAU CO THE DONG GOI SLOT COUNT: " << slot_count << endl; 

        cout << "3.1 MA HOA DATATRAIN. " << endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        auto[C_X_cols, C_Y_cols, C_W_col] = encrypt_x_y_w(num_feature, NUM_SAMPLES_TRAIN, num_label, scale, X_train, Y_train_onehot, encoder, encryptor);
        
        cout << "3.2 MA HOA DATATEST. " << endl;
        vector<vector<Ciphertext>> C_X_cols_test = encrypt_x(num_feature, num_label, scale, X_test, encoder, encryptor);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time_encrypt = chrono::duration_cast<chrono::milliseconds>(end1 - start1).count();


        print_example_banner("4. KHOI TAO VA MA HOA NGUONG BAN DAU.");
        
        cout << "4.1 KHOI TAO all_thresholds. " << endl;
        vector<double> all_thresholds;
        double min_T = -1.0;
        double max_T = 1.0;
        double step = (max_T - min_T -0.4) / (number_threshold + 1);
        int number_threshold_real = 0;
        for(int i=1 ; i<= number_threshold ; ++i){
            double theta = min_T + i * step;
            if(theta >= -0.2 && theta <= 0.2) continue; // bo qua vung giua
            if(theta >= max_T) break;
            all_thresholds.push_back(theta);
            number_threshold_real++;
        }

        cout << "4.2 MA HOA all_thresholds. " << endl; 
        auto start2 = std::chrono::high_resolution_clock::now();
        vector<Ciphertext> C_T_cols;
        for(double theta: all_thresholds){
            cout << "\tMa hoa theta = " << theta << endl;
            Plaintext c0_plain;
            encoder.encode(theta, scale, c0_plain);
            Ciphertext result;
            encryptor.encrypt(c0_plain, result); 
            C_T_cols.push_back(result);
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        double time_encrypt_thresholds = chrono::duration_cast<chrono::milliseconds>(end2 - start2).count();


        print_example_banner("5. Tree training.");

        cout << "5.1 MA HOA FEATURE DANG ONE-HOT DE PHUC VU CHO TINH TOAN." << endl;
        // // mã hóa số 0
        // Plaintext p_0;
        // encoder.encode(0.0, scale, p_0);
        // Ciphertext c_0;
        // encryptor.encrypt(p_0, c_0);
        // // mã hóa số 1
        // Plaintext p_1;
        // encoder.encode(1.0, scale, p_1);
        // Ciphertext c_1;
        // encryptor.encrypt(p_1, c_1);
        // vector<vector<Ciphertext>> C_I_one_hot; // size num_feature x num_feature mt duong cheo 
        // for (int i = 0; i < num_feature; ++i) {
        //     cout << "\tMA HOA ONE-HOT CHO feature thu " << i << endl;
        //     vector<Ciphertext> C_I_one_hot_f;
        //     C_I_one_hot_f.reserve(num_feature);
        //     for (int j = 0; j < num_feature; ++j) {
        //         if (j == i) {
        //             C_I_one_hot_f.push_back(c_1);  // 1 tại vị trí i
        //         } else {
        //             C_I_one_hot_f.push_back(c_0);  // 0 chỗ khác
        //         }
        //     }
        //     C_I_one_hot.push_back(std::move(C_I_one_hot_f));
        // }
        vector<Ciphertext> C_I_one_hot(num_feature);

        for (int i = 0; i < num_feature; ++i) {
            // 1. Tạo vector one-hot ở dạng plaintext
            vector<double> one_hot(num_feature, 0.0);
            one_hot[i] = 1.0;

            // 2. Encode
            Plaintext pt_one_hot;
            encoder.encode(one_hot, scale, pt_one_hot);

            // 3. Encrypt
            encryptor.encrypt(pt_one_hot, C_I_one_hot[i]);

            cout << "MA HOA ONE-HOT CHO feature thu " << i << endl;
        }


        cout << "5.2 HUAN LUYEN." << endl;
        auto start3 = std::chrono::high_resolution_clock::now();
        unique_ptr<Node_c> root = train_decision_tree(C_X_cols, C_W_col, C_Y_cols, C_T_cols, 0, max_depth, evaluator, encryptor, decryptor, encoder, relin_keys, gal_keys, context, num_feature, C_I_one_hot);
        auto end3 = std::chrono::high_resolution_clock::now();
        double time_training = chrono::duration_cast<chrono::milliseconds>(end3 - start3).count();
        
        // cout << "5.3 IN CAU TRUC CAY HOC DUOC (MA HOA)." << endl;
        // print_tree_structure(root_train);
        // cout << "5.4 IN CAU TRUC CAY HOC DUOC (GIAI MA)." << endl;
        // print_tree_decrypted(root_train, context, decryptor, encoder);

        auto start4 = std::chrono::high_resolution_clock::now();
        // cout << "5.3 SAVE MODEL TREE VAO " << src_model_tree << endl;
        // try{
        //     save_model(root_train, src_model_tree, num_feature);
        // }catch (const std::exception &ex) {
        //     std::cerr << "Error: " << ex.what() << std::endl;
        // }
        auto end4 = std::chrono::high_resolution_clock::now();
        double time_save_model = chrono::duration_cast<chrono::milliseconds>(end4 - start4).count();


        print_example_banner("6. PREDICT.");
        
        // cout << "6.1 LOAD MODEL TREE TU " << src_model_tree << endl;
        // auto start5 = std::chrono::high_resolution_clock::now();
        // unique_ptr<Node_c> root = load_model(src_model_tree , context, num_feature);
        // auto end5 = std::chrono::high_resolution_clock::now();
        // double time_load_model = chrono::duration_cast<chrono::milliseconds>(end5 - start5).count();

        // cout << "6.2 IN CAU TRUC CAY HOC DUOC SAU KHI LOAD (MA HOA)." << endl;
        // print_tree_structure(root);
        // cout << "6.3 IN CAU TRUC CAY HOC DUOC SAU KHI LOAD (GIAI MA)." << endl;
        // print_tree_decrypted(root, context, decryptor, encoder);

        cout << "6.2 DU DOAN TREN TAP TEST." << endl;
        auto start6 = std::chrono::high_resolution_clock::now();
        vector<vector<Ciphertext>> C_Final_Prediction;
        C_Final_Prediction.resize(num_label, vector<Ciphertext>(NUM_SAMPLES_TEST)); // num_label x N
        for(int i=0 ; i<NUM_SAMPLES_TEST ; i++){
            cout << "\tDu doan mau thu " << i << endl;
            vector<Ciphertext> C_Final_Prediction_c(num_label);

            C_Final_Prediction_c = predict_decision_tree(
                root, C_X_cols_test[i], evaluator, encryptor, encoder, 
                relin_keys, context, gal_keys, num_feature
            );
            
            for(int j=0 ; j<num_label ; j++){
                cout << "\t\tNhan thu " << j << endl;
                C_Final_Prediction[j][i] = C_Final_Prediction_c[j];
            }
        }
        auto end6 = std::chrono::high_resolution_clock::now();
        double time_predict_model = chrono::duration_cast<chrono::milliseconds>(end6 - start6).count();

        cout << "6.3 GIAI MA DU DOAN." << endl;
        vector<vector<double>> decoded_predictions; // LxN , L vector kich thuoc Nx1 
        decoded_predictions.resize(num_label, vector<double>(NUM_SAMPLES_TEST));
        for(int i=0 ; i<NUM_SAMPLES_TEST ; i++){
            for(int j=0 ; j<num_label ; j++){
                Plaintext pt_pred;
                decryptor.decrypt(C_Final_Prediction[j][i], pt_pred);
                vector<double> decoded_col;
                encoder.decode(pt_pred, decoded_col);
                double x = sum_slots(decoded_col, NUM_SAMPLES_TRAIN);
                decoded_predictions[j][i] = x;
            }
        }
        cout << "Ket qua du doan cho N mau (Lx1 vector trong moi slot):" << endl;
        cout << "Mau | Nhan     (0) | Nhan     (1) | Nhan Du Doan Cuoi" << endl;
        cout << "----|--------------|--------------| -------------------" << endl;

        for (size_t i = 0; i < NUM_SAMPLES_TEST; ++i) {
            double score_setosa = decoded_predictions[0][i]; // Lấy giá trị cột đầu tiên
            double score_versicolor = decoded_predictions[1][i]; // Lấy giá trị cột tiếp theo
            // Tạo vector để tìm max
            vector<double> scores = {score_setosa, score_versicolor};
            
            // Tìm điểm số lớn nhất
            double max_score = scores[0];
            size_t final_label = 0;

            for (size_t j = 1; j < num_label; ++j) {
                if (scores[j] > max_score) {
                    max_score = scores[j];
                    final_label = j;
                }
            }
            
            // In kết quả chi tiết
            cout << setw(3) << i << " | " 
                << setw(15) << fixed << setprecision(5) << score_setosa << " | " 
                << setw(19) << fixed << setprecision(5) << score_versicolor << " | " 
                << setw(16) << final_label << " (" << max_score << ")" << endl;
        }

        double acc = calculate_accuracy(decoded_predictions, Y_test_onehot);
        cout << "Accuracy = " << acc * 100.0 << "%\n";

        double f1_macro = calculate_f1_score(decoded_predictions, Y_test_onehot);
        cout << "F1-Macro Score = " << f1_macro << "\n";

        auto end = std::chrono::high_resolution_clock::now();

        long long start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();
        long long end_ms   = std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch()).count();

        // cout << "IN CAU TRUC CAY HOC DUOC SAU KHI LOAD (MA HOA)." << endl;
        // print_tree_structure(root);
        cout << "IN CAU TRUC CAY HOC DUOC SAU KHI LOAD (GIAI MA)." << endl;
        print_tree_decrypted(root, context, decryptor, encoder);

        save_log_json("C:/hu/decision-tree-he/save_info_model.csv", acc, NUM_SAMPLES_TRAIN, NUM_SAMPLES_TEST, src_data, number_threshold_real, max_depth, num_label, src_model_tree, time_encrypt, time_encrypt_thresholds, time_training, time_save_model, time_load_model, time_predict_model, start_ms, end_ms);
    } catch(const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    return 0;
}


