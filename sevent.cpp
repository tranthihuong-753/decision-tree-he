// five.cpp

// 1. Chuyển viec su dung node sang node_c ciphertext 

// 2. Tạo hàm multiply_ciphertexts() để nhân 2 ciphertext an toàn (align levels, scale, parms_id)
// Tao hàm add_ciphertexts() để cộng 2 ciphertext an toàn (align levels, scale, parms_id)
// Tao ham sub_ciphertexts() để trừ 2 ciphertext an toàn (align levels, scale, parms_id)

// 3. Tạo hàm save và load cho model tree dữ liệu mã hóa (trên chatgpt Linh có 1 bản)

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
using namespace std;
using namespace seal;

int number_threshold = 2; // Số ngưỡng để thử nghiệm trong train_decision_tree() thực tế số ngưỡng N x k = 70 x 4 = 280 
int max_depth = 2;
int NUM_SAMPLES_TRAIN = 8; // Số mẫu train trong tập dữ liệu iris.csv
int NUM_SAMPLES_TEST = 2;
size_t num_label = 3; // su dung cho ham predict_decision_tree()

string src_data = "C:/hu/decision-tree-he/Build_Project/Release/iris_s.csv";
size_t poly_modulus_degree = pow(2, 15); 
vector<int> coeff_modulus_bits = 
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40} ; // 18x40 
        // {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60} ;
        
double scale = pow(2.0, 40);

string src_model_tree = "model_tree_sevent.bin"; // C:\hu\decision-tree-he\Build_Project\model_tree_sevent.bin

// Giup one-hot encoding: "Iris-setosa" -> 0, "Iris-versicolor" -> 1, "Iris-virginica" -> 2
map<string, int> label_map = {
    {"setosa", 0},
    {"versicolor", 1},
    {"virginica", 2}
};

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

struct Sample {
    vector<double> features; 
    string label;           
};

vector<Sample> read_csv_dynamic(const std::string &filename) {
    std::cout << "read_csv_dynamic() " << filename << std::endl;
    std::vector<Sample> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Khong the mo file: " + filename);
    }

    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        Sample sample;

        std::vector<std::string> tokens;
        while (getline(ss, item, ',')) {
            tokens.push_back(item);
        }

        if (tokens.empty()) continue; // bo qua dong rong
        if (tokens.size() < 2) {
            throw std::runtime_error("Dong co du lieu khong hop le (it nhat 1 feature + 1 label).");
        }

        // Lấy tất cả cột trừ cột cuối cùng làm features
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            sample.features.push_back(std::stod(tokens[i]));
        }

        // Cột cuối cùng là label
        sample.label = tokens.back();
        data.push_back(sample);
    }

    std::cout << "compreted read_csv_dynamic() " << filename << std::endl;
    file.close();
    return data;
}

// Chuẩn hóa về khoảng [-1, 1]
void normalize(vector<Sample> &data)
{
    if (data.empty() || data[0].features.empty()) return;
    // size_t NUM_SAMPLES = data[0].features.size();
    for (int j = 0; j < NUM_SAMPLES_TRAIN; j++) {
        double min_val = 1e9, max_val = -1e9;
        for (auto &row : data) {
            min_val = min(min_val, row.features[j]);
            max_val = max(max_val, row.features[j]);
        }
        // Tránh chia cho 0
        if (abs(max_val - min_val) < 1e-12) continue;
        // Chuẩn hóa về khoảng [-1, 1]
        for (auto &row : data) {
            row.features[j] = 2 * (row.features[j] - min_val) / (max_val - min_val) - 1;
        }
    }
}

struct SplitData {
    vector<Sample> train, test;
};

// Chia dữ liệu thành train và test theo tỷ lệ train_ratio (0.8 = 80% train, 20% test) 
SplitData split_data(const std::vector<Sample>& data, double train_ratio = 0.8) {
    SplitData split;
    size_t n = data.size();
    size_t n_train = static_cast<size_t>(train_ratio * n);

    // Tạo danh sách index 0..n-1
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Chia train
    for (size_t i = 0; i < n_train; ++i)
        split.train.push_back(data[indices[i]]);

    // Chia test
    for (size_t i = n_train; i < n; ++i)
        split.test.push_back(data[indices[i]]);

    return split;
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

// ham nhan 2 ciphertext an toan 
Ciphertext multiply_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
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
    return result;
}

// ham cong an toan 
Ciphertext add_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const SEALContext &context
) {
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
    return result;
}

// ham tru an toan 
Ciphertext sub_ciphertexts(
    const Ciphertext &ct1,
    const Ciphertext &ct2,
    Evaluator &evaluator,
    const SEALContext &context
) {
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
    return result;
}

// leaf_value(W, Y) = sum(W * Y)
Ciphertext leaf_value(
    const Ciphertext &C_W_col,         
    const Ciphertext &C_Y_col,        
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const GaloisKeys &gal_keys,
    const SEALContext &context
){
    cout << "leaf_value()" << endl;

    // Ciphertext W = C_W_col;
    // Ciphertext Y = C_Y_col;

    // print_ct_info(context, W, "\tbefore align C_W_col");
    // print_ct_info(context, Y, "\tbefore align C_Y_col");

    // auto w_ci = context.get_context_data(W.parms_id())->chain_index();
    // auto y_ci = context.get_context_data(Y.parms_id())->chain_index();
    // if (w_ci > y_ci) {
    //     evaluator.mod_switch_to_inplace(W, Y.parms_id());
    //     cout << "mod-switched W down to Y's level" << endl;
    // } else if (y_ci > w_ci) {
    //     evaluator.mod_switch_to_inplace(Y, W.parms_id());
    //     cout << "mod-switched Y down to W's level" << endl;
    // }

    // print_ct_info(context, W, "\tafter align C_W_col");
    // print_ct_info(context, Y, "\tafter align C_Y_col");

    // size_t common_ci = context.get_context_data(W.parms_id())->chain_index();
    // if (common_ci < 1) {
    //     cerr << "leaf_value ERROR: insufficient levels (common chain_index < 1). Cannot multiply+rescale safely." << endl;
    //     throw runtime_error("leaf_value: insufficient levels for multiply");
    // }

    // Ciphertext P;
    // try {
    //     evaluator.multiply(W, Y, P);                 // <-- use W and Y, not C_W_col/C_Y_col
    //     evaluator.relinearize_inplace(P, relin_keys);
    //     evaluator.rescale_to_next_inplace(P);
    // } catch (const exception &e) {
    //     cerr << "leaf_value MULTIPLY/RESCALE exception: " << e.what() << endl;
    //     print_ct_info(context, W, "\tW (at failure)");
    //     print_ct_info(context, Y, "\tY (at failure)");
    //     throw;
    // }

    Ciphertext P = multiply_ciphertexts(
        C_W_col,
        C_Y_col,
        evaluator,
        relin_keys,
        context
    );

    Ciphertext res = P;
    for (int step = 1; step < NUM_SAMPLES_TRAIN; step+= 1) {
        Ciphertext tmp;
        evaluator.rotate_vector(res, step, gal_keys, tmp);
        evaluator.add_inplace(res, tmp);
    }

    cout << "Completed leaf_value() = " << endl;
    return res;
}

// tính z^i theo binary exponentiation để giảm độ ồn
Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    double scale,
    const SEALContext &context,
    const vector<double>& SOFT_STEP_COEFFICIENTS
) ///?
{
    cout << "soft_step_evaluation()" << endl;
    // 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(SOFT_STEP_COEFFICIENTS.size()); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // Tính lũy thừa của z với phương pháp bình phương và nhân (exponentiation by squaring) z^k với k ={1, 2, 4, 8}
    for(size_t i = 1; i < SOFT_STEP_COEFFICIENTS.size(); i *= 2) {
        // cout << " Computing power z^" << (i*2) << endl;
        Ciphertext z_for_mul = powers.back();
        // if (z_for_mul.parms_id() != powers.back().parms_id()) {
        //     evaluator.mod_switch_to_inplace(z_for_mul, powers.back().parms_id());
        // }

        // Ciphertext tmp;
        // try {
        //     evaluator.multiply(powers.back(), z_for_mul, tmp); // tmp scale ≈ s1 * s2
        // } catch (const exception &e) {
        //     cerr << "MULTIPLY EXCEPTION at i=" << i << ": " << e.what() << endl;
        //     throw;
        // }

        // evaluator.relinearize_inplace(tmp, relin_keys);
        // evaluator.rescale_to_next_inplace(tmp);

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
                    // auto temp_parms_id = temp_power.parms_id();
                    // if (temp_parms_id != powers[j].parms_id()) {
                    //     evaluator.mod_switch_to_inplace(temp_power, powers[j].parms_id());
                    // }
                    // evaluator.multiply_inplace(temp_power, powers[j]);
                    // evaluator.relinearize_inplace(temp_power, relin_keys);
                    // evaluator.rescale_to_next_inplace(temp_power);
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
        Ciphertext term;
        evaluator.multiply_plain(power, coeff_plain, term);
        evaluator.relinearize_inplace(term, relin_keys); //123
        evaluator.rescale_to_next_inplace(term);   

        auto target_parms_id = result.parms_id();
        if (term.parms_id() != target_parms_id) {
            evaluator.mod_switch_to_inplace(term, target_parms_id);
        }
        term.scale() = result.scale(); // Đảm bảo scale giống nhau trước khi cộng
        evaluator.add_inplace(result, term);
    }
    cout << "Completed soft_step_evaluation()" << endl;
    return result;
}

// cot x lay tu one-hot best_feature
Ciphertext compute_C_X_i(
    Ciphertext best_feature, // one-hot encoded
    const vector<Ciphertext>& C_X_cols,
    double scale,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    Evaluator& evaluator,
    const GaloisKeys& gal_keys,
    const RelinKeys& relin_keys,
    const SEALContext &context
)
{
    int num_features = C_X_cols.size();
    Ciphertext C_one_hot_best_feature = best_feature;

    // Tạo mask để giữ slot đầu tiên
    vector<double> mask_vec(encoder.slot_count(), 0.0);
    mask_vec[0] = 1.0;
    Plaintext pt_mask;
    encoder.encode(mask_vec, scale, pt_mask);

    Ciphertext C_X_i;
    for (int j = 0; j < num_features; ++j) {//
        Ciphertext rotated_one_hot_best_feature;
        evaluator.rotate_vector(C_one_hot_best_feature, static_cast<int>(j), gal_keys, rotated_one_hot_best_feature);

        // Nhân với mask
        Ciphertext C_masked;
        evaluator.multiply_plain(rotated_one_hot_best_feature, pt_mask, C_masked);
        evaluator.relinearize_inplace(C_masked, relin_keys);
        evaluator.rescale_to_next_inplace(C_masked);

        // nhan C_masked voi C_X_cols[j]
        // Ciphertext C_X_col_j = C_X_cols[j];
        // // align levels
        // auto c_ci = context.get_context_data(C_masked.parms_id())->chain_index();
        // auto x_ci = context.get_context_data(C_X_col_j.parms_id())->chain_index();
        // if (c_ci > x_ci) {
        //     evaluator.mod_switch_to_inplace(C_masked, C_X_col_j.parms_id());
        // } else if (x_ci > c_ci) {
        //     evaluator.mod_switch_to_inplace(C_X_col_j, C_masked.parms_id());
        // }
        // // multiply
        // evaluator.multiply_inplace(C_masked, C_X_col_j);
        // evaluator.relinearize_inplace(C_masked, relin_keys);
        // evaluator.rescale_to_next_inplace(C_masked);
        C_masked = multiply_ciphertexts(
            C_masked,
            C_X_cols[j],
            evaluator,
            relin_keys,
            context
        );
        // Cộng vào C_X_i
        if (j == 0) {
            C_X_i = C_masked;
        } else {
            // Đảm bảo cùng level và scale trước khi cộng
            // auto target_parms = C_X_i.parms_id();
            // if (C_masked.parms_id() != target_parms) {
            //     evaluator.mod_switch_to_inplace(C_masked, target_parms);
            // }
            // C_masked.scale() = C_X_i.scale();
            // evaluator.add_inplace(C_X_i, C_masked);
            C_X_i = add_ciphertexts(
                C_X_i,
                C_masked,
                evaluator,
                context
            );
        }
    }
    return C_X_i;
}

// compute_weighted_counts_homo(W, Y, X) = mul(W, soft_step_evaluation(cx[i]-theta), cyx)
pair<vector<Ciphertext>, vector<Ciphertext>> compute_weighted_counts_homo(
    Ciphertext best_feature, // 1 số plaintext 
    Ciphertext threshold, // 1 số plaintext 
    const vector<Ciphertext>& C_X_cols, // K ciphertext
    const Ciphertext& C_W_col,          // 1 ciphertextMULTIPLY EXCEPTION
    const vector<Ciphertext>& C_Y_cols,   // L ciphertext
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& gal_keys,
    const SEALContext& context,
    double scale // tính soft_step_evaluation()
    // int NUM_SAMPLES 
)
{
    cout << "compute_weighted_counts_homo()" << endl;
        
    // Helper: align levels then multiply (safe)
    // auto align_and_multiply = [&](const Ciphertext &A, const Ciphertext &B, Ciphertext &Out) {
    //     // Make copies so originals are not mutated
    //     Ciphertext a = A;
    //     Ciphertext b = B;

    //     auto a_ci = context.get_context_data(a.parms_id())->chain_index();
    //     auto b_ci = context.get_context_data(b.parms_id())->chain_index();
    //     // cout << "  align_and_multiply: a.chain_index=" << a_ci << ", b.chain_index=" << b_ci << endl;

    //     // If levels differ, mod-switch the one on higher level down to the other's level
    //     if (a.parms_id() != b.parms_id()) {
    //         if (a_ci > b_ci) {
    //             evaluator.mod_switch_to_inplace(a, b.parms_id());
    //             // cout << "   mod-switched a down to b's level" << endl;
    //         } else {
    //             evaluator.mod_switch_to_inplace(b, a.parms_id());
    //             // cout << "   mod-switched b down to a's level" << endl;
    //         }
    //     }

    //     // After alignment, get the (common) chain index
    //     auto common_ci = context.get_context_data(a.parms_id())->chain_index();
    //     // cout << "   common chain_index=" << common_ci << endl;

    //     // Need at least chain_index >= 1 to allow rescale_to_next after multiply
    //     if (common_ci < 1) {
    //         cerr << "   ERROR: insufficient levels (chain_index < 1). Cannot multiply-and-rescale." << endl;
    //         throw runtime_error("Insufficient levels for multiply");
    //     }

    //     // cout << "   scales before mul: a.scale=" << a.scale() << ", b.scale=" << b.scale() << endl;

    //     try {
    //         evaluator.multiply(a, b, Out);
    //         evaluator.relinearize_inplace(Out, relin_keys);
    //         evaluator.rescale_to_next_inplace(Out);
    //     } catch (const exception &e) {
    //         cerr << "   MULTIPLY EXCEPTION: " << e.what() << endl;
    //         print_ct_info(context, a, "   a (after align)");
    //         print_ct_info(context, b, "   b (after align)");
    //         throw;
    //     }
    // };

    // cout << "compute_weighted_counts_homo()" << endl;
    const size_t NUM_LABELS = C_Y_cols.size(); // 3 
    // cout << "NUM_LABELS=" << NUM_LABELS << endl; //3 
    // Khởi tạo vector kết quả Lx1 (một ciphertext cho mỗi nhãn)
    vector<Ciphertext> C_right_counts(NUM_LABELS);
    vector<Ciphertext> C_left_counts(NUM_LABELS);
    
    // --- BƯỚC 1: TÍNH SOFT-STEP VÀ TRỌNG SỐ TẠI SLOT (W_Phi) ---
    Ciphertext C_X_i = compute_C_X_i(
        best_feature, C_X_cols, scale, encryptor, encoder, evaluator, gal_keys, relin_keys, context
    );
    
    auto target_parms = C_X_i.parms_id();
    double target_scale = C_X_i.scale();

    // Mã hóa Ngưỡng Theta
    Ciphertext C_Theta = threshold;
    if (C_Theta.parms_id() != target_parms) {
        evaluator.mod_switch_to_inplace(C_Theta, target_parms);
    }
    C_Theta.scale() = target_scale;

    // Độ lệch Z = X[best_feature] - theta
    Ciphertext C_Z_right, C_Z_left; 
    try{
        evaluator.sub(C_X_i, C_Theta, C_Z_right); 
        evaluator.sub(C_Theta, C_X_i, C_Z_left); 
    } catch (const exception &e) {
        cerr << "SUBTRACTION EXCEPTION in compute_W_phi_best: " << e.what()
                << ". Check levels of C_X_i and C_Theta." << endl;
        throw;
    }

    // cout << "Tinh Soft-Step" << endl;
    Ciphertext C_Phi_Right = soft_step_evaluation(
        C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    Ciphertext C_Phi_Left = soft_step_evaluation(
        C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16

    // Tính W_phi = W * Phi
    // cout << "Tinh W_Phi" << endl;
    Ciphertext C_W_Phi_Right, C_W_Phi_Left;
    
    // Right: C_W_col * C_Phi_Right
    // cout << " Right: C_W_col * C_Phi_Right" << endl;
    // align_and_multiply(C_W_col, C_Phi_Right, C_W_Phi_Right);
    C_W_Phi_Right = multiply_ciphertexts(
        C_W_col,
        C_Phi_Right,
        evaluator,
        relin_keys,
        context
    );
    // cout << "Right: C_W_col * C_Phi_Right: ";
    // int i;
    // for (i = 0; i < 8; i++) {
    //     cout << values_1[i] << "    ";
    // }
    // cout << endl;

    // Left: C_W_col * C_Phi_Left
    // cout << " Left: C_W_col * C_Phi_Left" << endl;
    // align_and_multiply(C_W_col, C_Phi_Left, C_W_Phi_Left);
    C_W_Phi_Left = multiply_ciphertexts(
        C_W_col,
        C_Phi_Left,
        evaluator,
        relin_keys,
        context
    );
    // vector<double> values_1 = decrypt_to_vector(C_W_Phi_Left, decryptor, encoder);
    // cout << "C_W_Phi_Left: ";
    // int i;
    // for (i = 0; i < 8; i++) {
    //     cout << values_1[i] << " ";
    // }
    // cout << endl;


    // Tính W.Phi.Y và tổng trọng số cho 1 nhãn bị chia bởi i, theta 
    for (size_t l = 0; l < NUM_LABELS; ++l) {
        // cout << "Processing label " << l << " (TRY best multiply order)" << endl;
        Ciphertext C_Term_Right_tmp;
        // Strategy: try (W*Y) then *Phi  OR (W*Phi) then *Y  OR (Phi*Y) then *W
        bool done = false;
        vector<string> attempts = {"(W*Y)*Phi", "(W*Phi)*Y", "(Phi*Y)*W"};
        for (auto &attempt : attempts) {
            try {
                // cout << " Attempting order: " << attempt << endl;
                if (attempt == "(W*Y)*Phi") {
                    Ciphertext t1;
                    // align_and_multiply(C_W_col, C_Y_cols[l], t1);   
                    t1 = multiply_ciphertexts(
                        C_W_col,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                    // align_and_multiply(t1, C_Phi_Right, C_Term_Right_tmp);
                    C_Term_Right_tmp = multiply_ciphertexts(
                        t1,
                        C_Phi_Right,
                        evaluator,
                        relin_keys,
                        context
                    ); 
                } else if (attempt == "(W*Phi)*Y") {
                    // align_and_multiply(C_W_Phi_Right, C_Y_cols[l], C_Term_Right_tmp);
                    C_Term_Right_tmp = multiply_ciphertexts(
                        C_W_Phi_Right,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                } else { // (Phi*Y)*W
                    Ciphertext t1;
                    // align_and_multiply(C_Phi_Right, C_Y_cols[l], t1);
                    t1 = multiply_ciphertexts(
                        C_Phi_Right,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                    // align_and_multiply(t1, C_W_col, C_Term_Right_tmp);
                    C_Term_Right_tmp = multiply_ciphertexts(
                        t1,
                        C_W_col,
                        evaluator,
                        relin_keys,
                        context
                    );
                }
                C_right_counts[l] = C_Term_Right_tmp;
                done = true;
                // vector<double> values_1 = decrypt_to_vector(C_right_counts[l], decryptor, encoder);
                // cout << "(Phi*Y)*W right: " << l ;
                // int i;
                // for (i = 0; i < NUM_SAMPLES + 1; i++) {
                //     cout << values_1[i] << "    ";
                // }
                // cout << endl;
                break;
            } catch (const exception &e) {
                cout << "  Attempt " << attempt << " failed: " << e.what() << endl;
            }
        }
        
        if (!done) {
            // None of the orders worked: we are out of levels.
            cerr << "ERROR: cannot compute W*Phi*Y for label " << l 
                << " with current levels. Consider increasing coeff_modulus or reducing operations." << endl;
            // throw or handle gracefully; here we throw to stop and show debug
            throw runtime_error("compute_weighted_counts_homo: insufficient levels for W*Phi*Y");
        }

        Ciphertext C_Sum_Right_l = C_right_counts[l];
        for (int step = 1; step < NUM_SAMPLES_TRAIN ; step += 1) {
            Ciphertext C_Rotated;
            evaluator.rotate_vector(C_Sum_Right_l, step, gal_keys, C_Rotated);
            evaluator.add_inplace(C_Sum_Right_l, C_Rotated);
        }
        C_right_counts[l] = C_Sum_Right_l;

        // vector<double> values_1 = decrypt_to_vector(C_right_counts[l], decryptor, encoder);
        // cout << "nhan " << l ;
        // int i;
        // for (i = 0; i < NUM_SAMPLES + 1; i++) {
        //     cout << values_1[i] << "    ";
        // }
        // cout << endl;
        
    }

    for (size_t l = 0; l < NUM_LABELS; ++l){
        Ciphertext C_Term_Left_tmp;
        vector<string> left_attempts = {"(W*Y)*Phi", "(W*Phi)*Y", "(Phi*Y)*W"};
        bool done_left = false;
        for (auto &attempt : left_attempts) {
            try {
                if (attempt == "(W*Y)*Phi") {
                    Ciphertext t1;
                    // align_and_multiply(C_W_col, C_Y_cols[l], t1);
                    t1 = multiply_ciphertexts(
                        C_W_col,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                    // align_and_multiply(t1, C_Phi_Left, C_Term_Left_tmp);
                    C_Term_Left_tmp = multiply_ciphertexts(
                        t1,
                        C_Phi_Left,
                        evaluator,
                        relin_keys,
                        context
                    );
                } else if (attempt == "(W*Phi)*Y") {
                    // align_and_multiply(C_W_Phi_Left, C_Y_cols[l], C_Term_Left_tmp);
                    C_Term_Left_tmp = multiply_ciphertexts(
                        C_W_Phi_Left,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                } else { // (Phi*Y)*W
                    Ciphertext t1;
                    // align_and_multiply(C_Phi_Left, C_Y_cols[l], t1);
                    t1 = multiply_ciphertexts(
                        C_Phi_Left,
                        C_Y_cols[l],
                        evaluator,
                        relin_keys,
                        context
                    );
                    // align_and_multiply(t1, C_W_col, C_Term_Left_tmp);
                    C_Term_Left_tmp = multiply_ciphertexts(
                        t1,
                        C_W_col,
                        evaluator,
                        relin_keys,
                        context
                    );
                }

                C_left_counts[l] = C_Term_Left_tmp;
                done_left = true;
                break;
            } catch (const exception &e) {
                cout << "  LEFT attempt " << attempt << " failed: " << e.what() << endl;
            }
        }

        // --- SUM ROTATIONS for LEFT (giống RIGHT) ---
        Ciphertext C_Sum_Left_l = C_left_counts[l];
        for (int step = 1; step < NUM_LABELS ; step += 1) {
            Ciphertext C_Rotated;
            evaluator.rotate_vector(C_Sum_Left_l, step, gal_keys, C_Rotated);
            evaluator.add_inplace(C_Sum_Left_l, C_Rotated);
        }
        C_left_counts[l] = C_Sum_Left_l; // tong bang so phan tu dau tien
    }

    cout << "Completed compute_weighted_counts_homo()" << endl;
    return {C_right_counts, C_left_counts}; // vecto<cipher=vecto Nx1>
}

struct Node {
    bool is_leaf = false;
    // Giá trị nút lá (Nếu là lá, kích thước Lx1) leaf_value[l] = trọng số chỉ xác suất rơi vào nhãn l  
    vector<double> leaf_value; 

    // Thông số phân chia (Nếu không là lá)
    int feature_index = -1;
    double threshold = 0.0; 

    // Các nhánh (Node con)
    unique_ptr<Node> left_child = nullptr;
    unique_ptr<Node> right_child = nullptr;
};

//1
struct Node_c {
    bool is_leaf = false;
    // Giá trị nút lá (Nếu là lá, kích thước Lx1) leaf_value[l] = trọng số chỉ xác suất rơi vào nhãn l  
    Ciphertext leaf_value; 

    // Thông số phân chia (Nếu không là lá)
    Ciphertext feature_index; // one-hot 
    Ciphertext threshold; // so 

    // Các nhánh (Node con)
    unique_ptr<Node_c> left_child = nullptr;
    unique_ptr<Node_c> right_child = nullptr;
};

// compute_gini_impurity() Tính độ tạp Gini (Gini Impurity) cho một phân chia cụ thể
double compute_gini_impurity(
    const vector<double>& right_counts, // Vector Lx1 cleartext (tổng trọng số bên phải)
    const vector<double>& left_counts
)  // Vector Lx1 cleartext (tổng trọng số bên trái)
{
    cout << "compute_gini_impurity()" << endl;
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
    Ciphertext best_feature, Ciphertext best_threshold,
    const vector<Ciphertext>& C_X_cols, const Ciphertext& C_W_col, 
    Evaluator& evaluator, Encryptor& encryptor, const CKKSEncoder& encoder,
    const RelinKeys& relin_keys, double scale, const SEALContext& context, const GaloisKeys& gal_keys)
{
    cout << "compute_W_phi_best()" << endl;
    // auto align_and_multiply = [&](const Ciphertext& a, const Ciphertext& b, Ciphertext& result) {
    //     Ciphertext a_copy = a;
    //     Ciphertext b_copy = b;

    //     auto a_level = context.get_context_data(a_copy.parms_id())->chain_index();
    //     auto b_level = context.get_context_data(b_copy.parms_id())->chain_index();

    //     if (a_level > b_level) {
    //         evaluator.mod_switch_to_inplace(a_copy, b_copy.parms_id());
    //     } else if (b_level > a_level) {
    //         evaluator.mod_switch_to_inplace(b_copy, a_copy.parms_id());
    //     }

    //     evaluator.multiply(a_copy, b_copy, result);
    //     evaluator.relinearize_inplace(result, relin_keys);
    //     evaluator.rescale_to_next_inplace(result);
    // };

    // Mã hóa one-hot best_feature ???
    Ciphertext C_X_i = compute_C_X_i(
        best_feature, C_X_cols, scale, encryptor, encoder, evaluator, gal_keys, relin_keys, context
    );
    
    auto target_parms = C_X_i.parms_id();
    double target_scale = C_X_i.scale();

    // Mã hóa Ngưỡng Theta
    Ciphertext C_Theta = best_threshold;
    if (C_Theta.parms_id() != target_parms) {
        evaluator.mod_switch_to_inplace(C_Theta, target_parms);
    }
    C_Theta.scale() = target_scale;

    // Độ lệch Z = X[best_feature] - theta
    Ciphertext C_Z_right, C_Z_left; 
    try{
        evaluator.sub(C_X_i, C_Theta, C_Z_right); 
        evaluator.sub(C_Theta, C_X_i, C_Z_left); 
    } catch (const exception &e) {
        cerr << "SUBTRACTION EXCEPTION in compute_W_phi_best: " << e.what()
                << ". Check levels of C_X_i and C_Theta." << endl;
        throw;
    }

    // Soft-Step
    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); //?SOFT_STEP_COEFFICIENTS_16
    
    // W_new = W * Phi
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

//2 
unique_ptr<Node_c> train_decision_tree(
    const vector<Ciphertext>& C_X_cols,
    const Ciphertext& C_W_col, 
    const vector<Ciphertext>& C_Y_cols,
    const vector<double>& all_thresholds, // Tập hợp các ngưỡng duy nhất
    int depth,
    int max_depth,
    Evaluator& evaluator,
    Encryptor& encryptor,
    Decryptor& decryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& gal_keys,
    const SEALContext& context,
    double scale
    // size_t NUM_SAMPLES
){
    cout << "train_decision_tree() depth=" << depth << endl;
    auto node = make_unique<Node>();
    const size_t NUM_FEATURES = C_X_cols.size();
    size_t NUM_LABELS = C_Y_cols.size();

    //2 
    auto node_c = make_unique<Node_c>();

    // 1. ĐIỀU KIỆN DỪNG
    if (depth >= max_depth) {
        // cout << " Reached max depth. Creating leaf node." << endl;
        // node->is_leaf = true;
        // node->leaf_value.resize(NUM_LABELS); // Khởi tạo vector kết quả Lx1
        //3 
        node_c->is_leaf = true;

        // Lặp qua TẤT CẢ L nhãn để tính tổng trọng số cho từng nhãn
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            // cout << "  Computing leaf value for label " << l << endl;
            // 1.1. TÍNH TỔNG TRỌNG SỐ ĐỒNG HÌNH (C_W * C_Y[l])
            Ciphertext C_weighted_sum = leaf_value(
                C_W_col, C_Y_cols[l], evaluator, relin_keys, gal_keys, context 
            );

            //4 
            node_c->leaf_value = C_weighted_sum;

            // 1.2. GIẢI MÃ (Client-side)
            // Plaintext pt_result;
            // decryptor.decrypt(C_weighted_sum, pt_result);
            
            // vector<double> decoded_result;
            // encoder.decode(pt_result, decoded_result);
            
            // // 1.3. LƯU GIÁ TRỊ LÁ (Slot 0 chứa tổng cuối cùng)
            // node->leaf_value[l] = decoded_result[0];
        }
        
        // cout << "Created leaf node at depth=" << depth << " with values: ";
        return node_c; 
    }

    // 2. TÌM NGƯỠNG TỐI ƯU
    double min_gini = 1e9;
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // (Vector này sẽ lưu các giá trị W*phi(Z) cho lần gọi đệ quy sau)
    Ciphertext C_W_Phi_Best_Right, C_W_Phi_Best_Left;

    // Lặp qua TẤT CẢ thuộc tính (i) và TẤT CẢ ngưỡng (theta)
    for (int i = 0; i < NUM_FEATURES; ++i) { // i < NUM_FEATURES
        // ma hoa one-hot feature i ???
        vector<double> one_hot_vec(encoder.slot_count(), 0.0);
        one_hot_vec[i] = 1;
        Plaintext pt_one_hot;
        encoder.encode(one_hot_vec, scale, pt_one_hot);
        Ciphertext C_one_hot_feature;
        encryptor.encrypt(pt_one_hot, C_one_hot_feature);

        // cout << " Feature " << i << endl;
        for (double threshold : all_thresholds) { // 163 ngưỡng // double threshold : all_thresholds
            // cout << "  Threshold " << threshold << endl;
            // ma hoa threshold
            Plaintext pt_threshold;
            encoder.encode(threshold, scale, pt_threshold);
            Ciphertext C_threshold;
            encryptor.encrypt(pt_threshold, C_threshold);

            // 2.1. Tính tổng trọng số bảo mật (Homomorphic Counts)
            auto [C_right_counts, C_left_counts] = compute_weighted_counts_homo(
                C_one_hot_feature, C_threshold, C_X_cols, C_W_col, C_Y_cols, 
                evaluator, encryptor, encoder, relin_keys, gal_keys, context, scale
            );
            
            // 2.2. Giải mã và Tính Gini (Client-side)
            vector<double> right_counts_clear(NUM_LABELS);
            vector<double> left_counts_clear(NUM_LABELS);

            for (size_t l = 0; l < NUM_LABELS; ++l) {
                // Giải mã và lấy giá trị từ slot 0
                Plaintext pt_r, pt_l;
                decryptor.decrypt(C_right_counts[l], pt_r);
                decryptor.decrypt(C_left_counts[l], pt_l);
                
                vector<double> decoded_r, decoded_l;
                encoder.decode(pt_r, decoded_r);
                encoder.decode(pt_l, decoded_l);
                
                right_counts_clear[l] = decoded_r[0]; 
                left_counts_clear[l] = decoded_l[0];
            }
            
            // 2.3. Tính Gini Impurity (Cleartext)
            double current_gini = compute_gini_impurity(right_counts_clear, left_counts_clear);

            // 2.4. Cập nhật ngưỡng tốt nhất
            if (current_gini < min_gini) {
                min_gini = current_gini;
                best_feature = i;
                best_threshold = threshold;       
            }
        }

    }
    
    // 3. THIẾT LẬP THÔNG SỐ NÚT VÀ GỌI ĐỆ QUY
    // node->feature_index = best_feature;
    // node->threshold = best_threshold;

    //5
    // Chuyen best_feature thành ciphertext one-hot
    Plaintext pt_best_feature;
    encoder.encode(static_cast<double>(best_feature), scale, pt_best_feature);
    Ciphertext C_best_feature;
    encryptor.encrypt(pt_best_feature, C_best_feature);
    // ma hoa best_threshold
    Plaintext pt_best_threshold;
    encoder.encode(best_threshold, scale, pt_best_threshold);
    Ciphertext C_best_threshold;
    encryptor.encrypt(pt_best_threshold, C_best_threshold);
    node_c->feature_index = C_best_feature; // one-hot
    node_c->threshold = C_best_threshold; // so 

    // TÍNH W_NEW CHO ĐỆ QUY 
    auto [C_W_new_right, C_W_new_left] = compute_W_phi_best(
        C_best_feature, C_best_threshold, C_X_cols, C_W_col, 
        evaluator, encryptor, encoder, relin_keys, scale, context, gal_keys
    );

    // 4. GỌI ĐỆ QUY (RECURSION)
    node_c->right_child = train_decision_tree(
        C_X_cols, C_W_new_right, C_Y_cols, all_thresholds, 
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, gal_keys, context, scale
    );
    node_c->left_child = train_decision_tree(
        C_X_cols, C_W_new_left, C_Y_cols, all_thresholds,
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, gal_keys, context, scale
    );

    cout << "Completed train_decision_tree() at depth=" << depth << endl;
    return node_c;
}

void save_ciphertext(const Ciphertext &ct, ostream &os)
{
    cout << "save ct size=" << ct.save_size() << endl;
    ct.save(os); // save trực tiếp vào stream
}

Ciphertext load_ciphertext(istream &is, const SEALContext &context)
{
    cout << "[load ct] pos=" << is.tellg() << endl;
    Ciphertext ct;
    ct.load(context, is);
    cout << "[load ct] done pos=" << is.tellg() << endl;
    return ct;
}

void save_node(const unique_ptr<Node_c> &node, ostream &os)
{
    bool exist = (node != nullptr);
    os.write((char*)&exist, sizeof(exist));
    if (!exist) return;

    os.write((char*)&node->is_leaf, sizeof(node->is_leaf));

    if (node->is_leaf)
    {
        // chỉ save leaf_value
        save_ciphertext(node->leaf_value, os);

        // không có child
        bool has_left = false, has_right = false;
        os.write((char*)&has_left, sizeof(has_left));
        os.write((char*)&has_right, sizeof(has_right));
        return;
    }

    // node split
    save_ciphertext(node->feature_index, os);
    save_ciphertext(node->threshold, os);
    // save_ciphertext(node->leaf_value, os); // có thể giữ leaf_value nếu bạn dùng để store distribution

    bool has_left = (node->left_child != nullptr);
    bool has_right = (node->right_child != nullptr);
    os.write((char*)&has_left, sizeof(has_left));
    os.write((char*)&has_right, sizeof(has_right));

    if (has_left) save_node(node->left_child, os);
    if (has_right) save_node(node->right_child, os);
}

unique_ptr<Node_c> load_node(istream &is, const SEALContext &context)
{
    cout << "load_node()" << endl;
    bool exist;
    is.read((char*)&exist, sizeof(exist));
    if (!exist) return nullptr;

    auto node = make_unique<Node_c>();

    is.read((char*)&node->is_leaf, sizeof(node->is_leaf));

    if (node->is_leaf)
    {
        cout << "is_leaf" << endl;
        node->leaf_value = load_ciphertext(is, context);

        bool has_left, has_right;
        is.read((char*)&has_left, sizeof(has_left));
        is.read((char*)&has_right, sizeof(has_right));

        return node;
    }

    // Non-leaf
    cout << "non-leaf: load feature_index" << endl;
    node->feature_index = load_ciphertext(is, context);
    cout << "non-leaf: load threshold" << endl;
    node->threshold = load_ciphertext(is, context);
    cout << "3" << endl;
    // node->leaf_value = load_ciphertext(is, context);

    bool has_left, has_right;
    cout << "4" << endl;
    is.read((char*)&has_left, sizeof(has_left));
    cout << "5" << endl;
    is.read((char*)&has_right, sizeof(has_right));

    cout << "has_left=" << has_left << " has_right=" << has_right << endl;
    if (has_left) node->left_child = load_node(is, context);
    if (has_right) node->right_child = load_node(is, context);

    return node;
}

void save_model(const unique_ptr<Node_c> &root, const string &filepath)
{
    cout << "save_model()" << endl;
    ofstream ofs(filepath, ios::binary);
    if (!ofs) throw runtime_error("Cannot open model file to save!");
    save_node(root, ofs);
    cout << "completed save_model()" << endl;
}

unique_ptr<Node_c> load_model(const string &filepath, const SEALContext &context)
{
    cout << "load_model()" << endl;
    ifstream ifs(filepath, ios::binary);
    if (!ifs) throw runtime_error("Cannot open model file to load!");
    return load_node(ifs, context);
}

vector<Ciphertext> predict_decision_tree(
    const unique_ptr<Node_c>& node_c,       // Nút hiện tại của cây
    const vector<Ciphertext>& C_X_cols, // K ciphertext (Dữ liệu đầu vào, Column Batched)
    const Ciphertext& C_W_current,      // Trọng số hiện tại của mẫu (W * phi * phi...)
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const SEALContext& context,
    double scale,
    const GaloisKeys& galois_keys
)
{
    cout << "predict_decision_tree()" << endl;

    // NÚT LÁ
    if (node_c->is_leaf) {
        cout << " Reached leaf node. Processing leaf values." << endl;
        const size_t NUM_LABELS = num_label;

        vector<Ciphertext> C_Leaf_Output;
        C_Leaf_Output.reserve(NUM_LABELS);


        for (size_t l = 0; l < NUM_LABELS; ++l) {
            Ciphertext C_leaf_value = node_c->leaf_value; // Lấy ciphertext leaf_value từ nút
            Ciphertext C_W_current_copy = C_W_current;
            // 2) Mod-switch C_leaf_value xuống cùng parms_id với C_W_current
            auto data_leaf = context.get_context_data(C_leaf_value.parms_id());
            size_t leaf_level = data_leaf->chain_index();
            auto data_w = context.get_context_data(C_W_current.parms_id());
            size_t w_level = data_w->chain_index();
            if (leaf_level > w_level) 
                evaluator.mod_switch_to_inplace(C_leaf_value, C_W_current.parms_id());
            else if (w_level > leaf_level) 
                evaluator.mod_switch_to_inplace(C_W_current_copy, C_leaf_value.parms_id());

            // 3) Multiply_plain: C_W_current * C_leaf_value
            Ciphertext C_Leaf_l;
            evaluator.multiply(C_W_current_copy, C_leaf_value, C_Leaf_l);

            // 4) Relinearize và rescale (nếu cần)
            evaluator.relinearize_inplace(C_Leaf_l, relin_keys);

            // Rescale nếu scale quá lớn (bạn có thể giữ nguyên policy rescale như code gốc)
            try {
                evaluator.rescale_to_next_inplace(C_Leaf_l);
            } catch (const exception &e) {
                // Nếu rescale fail (ví dụ ở cuối chain), bạn có thể bỏ qua hoặc điều chỉnh
                cerr << "  Warning: rescale failed on leaf label " << l << " : " << e.what() << endl;
            }

            C_Leaf_Output.push_back(std::move(C_Leaf_l));
            cout << " Processing leaf label " << l << endl;
        }
        return C_Leaf_Output;
    }

    // KHÔNG PHẢI LÁ — TÍNH SOFT-STEP
    Ciphertext i_best = node_c->feature_index;
    Ciphertext C_Theta = node_c->threshold;

    const Ciphertext& C_X_i = compute_C_X_i(
        i_best, C_X_cols, scale, encryptor, encoder, evaluator, galois_keys, relin_keys, context
    );

    Ciphertext C_Z_right, C_Z_left;
    // evaluator.sub(C_X_i, C_Theta, C_Z_right);
    C_Z_right = sub_ciphertexts(C_X_i, C_Theta, evaluator, context);
    // evaluator.sub(C_Theta, C_X_i, C_Z_left);
    C_Z_left = sub_ciphertexts(C_Theta, C_X_i, evaluator, context);

    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_8);
    Ciphertext C_Phi_Left  = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_8);

    Ciphertext C_W_Right, C_W_Left;
    // cout << " Multiplying W_current * Phi_Right..." << endl;
    // safe_multiply(C_W_current, C_Phi_Right, C_W_Right, "Right");
    C_W_Right = multiply_ciphertexts(
        C_W_current,
        C_Phi_Right,
        evaluator,
        relin_keys,
        context
    );

    // cout << " Multiplying W_current * Phi_Left..." << endl;
    // safe_multiply(C_W_current, C_Phi_Left, C_W_Left, "Left");
    C_W_Left = multiply_ciphertexts(
        C_W_current,
        C_Phi_Left,
        evaluator,
        relin_keys,
        context
    );

    // GỌI ĐỆ QUY
    vector<Ciphertext> C_Output_Right = predict_decision_tree(
        node_c->right_child, C_X_cols, C_W_Right, evaluator, encryptor, encoder, 
        relin_keys, context, scale, galois_keys
    );

    vector<Ciphertext> C_Output_Left = predict_decision_tree(
        node_c->left_child, C_X_cols, C_W_Left, evaluator, encryptor, encoder, 
        relin_keys, context, scale, galois_keys
    );

    // if (C_Output_Right.parms_id() != C_Output_Left.parms_id()) {
    //     auto ci_r = context.get_context_data(C_Output_Right.parms_id())->chain_index();
    //     auto ci_l = context.get_context_data(C_Output_Left.parms_id())->chain_index();
    //     if (ci_r > ci_l)
    //         evaluator.mod_switch_to_inplace(C_Output_Right, C_Output_Left.parms_id());
    //     else
    //         evaluator.mod_switch_to_inplace(C_Output_Left, C_Output_Right.parms_id());
    // }

    vector<Ciphertext> C_Final_Output;
    size_t NUM_LABELS = C_Output_Right.size();
    for (size_t l = 0; l < NUM_LABELS; ++l) {
        // if (C_Output_Right[l].parms_id() != C_Output_Left[l].parms_id()) {
        //     auto ci_r = context.get_context_data(C_Output_Right[l].parms_id())->chain_index();
        //     auto ci_l = context.get_context_data(C_Output_Left[l].parms_id())->chain_index();
        //     if (ci_r > ci_l)
        //         evaluator.mod_switch_to_inplace(C_Output_Right[l], C_Output_Left[l].parms_id());
        //     else
        //         evaluator.mod_switch_to_inplace(C_Output_Left[l], C_Output_Right[l].parms_id());
        // }
        // Ciphertext C_Sum;
        // evaluator.add(C_Output_Right[l], C_Output_Left[l], C_Sum);
        Ciphertext C_Sum = add_ciphertexts(
            C_Output_Right[l],
            C_Output_Left[l],
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


int main()
{
    print_example_banner("1. Thiet lap tham so CKKS");
    cout << "Khoi tao tham so (poly_modulus_degree va coeff_modulus_bits va scale)" << endl;
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

    print_example_banner("2. Tao khoa va cac doi tuong SEAL");
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

    print_example_banner("3. Doc du lieu");
    vector<Sample> data;
    try {
        data = read_csv_dynamic(src_data);
        if (data.empty()) throw runtime_error("Khong doc duoc du lieu.");
        normalize(data);
    } catch (const exception &e) {
        cerr << "Loi: " << e.what() << endl;
        return 1;
    }

    // 4. Xử lý dữ liệu 
    print_example_banner("4. Xu ly du lieu");
    SplitData split = split_data(data);

    const size_t NUM_LABELS = label_map.size(); 
    cout << "So nhan (L) NUM_LABELS = " << NUM_LABELS << endl; 
    
    // Trích xuất dữ liệu từ SplitData
    auto extract_data = [&](const vector<Sample>& iris_data, vector<vector<double>>& X, vector<vector<double>>& Y_onehot) {
        if (iris_data.empty()) return;
        
        int iii = 0;
        for (const auto& iris : iris_data) {
            // X
            X.push_back(iris.features);
            // cout << iii << " / " << iris.features[0] << iris.features[1] << iris.features[2] << iris.features[3] << iris.features[4] << iris.features[5] << iris.features[6] << iris.features[7] << endl;

            // Y (One-Hot Vector - Kích thước Lx1, ví dụ: 3x1)
            vector<double> y_onehot(NUM_LABELS, 0.0); // 3x1 khởi tạo 0.0
            // Tìm index của nhãn và đặt giá trị 1.0
            if (label_map.count(iris.label)) {
                int label_index = label_map.at(iris.label);
                y_onehot[label_index] = 1.0;
            } else {
                // Xử lý lỗi nếu nhãn không hợp lệ (nên thêm logic log lỗi)
                cerr << "Canh bao: Nhãn khong hop le: " << iris.label << endl;
            }
            Y_onehot.push_back(y_onehot);
            // cout << iii << " / " << iris.label << " kich thuoc " << y_onehot.size() << " / " << y_onehot[0] << y_onehot[1] << y_onehot[2] << endl;
            iii++;
        }
        // cout <<  "Kich thuoc ma tran Y_onehot: " << Y_onehot.size() << " x " << Y_onehot[0].size() << endl;
    };

    vector<vector<double>> X_train, Y_train_onehot;
    extract_data(split.train, X_train, Y_train_onehot); // X_train là ma trận NxK (70x7) ,Y_train_onehot là ma trận N x L (70x3)
    // cout << "Kich thuoc Y_train_onehot: " << Y_train_onehot.size() << "x" << Y_train_onehot[0].size() << endl; // 70x3 

    const size_t NUM_SAMPLES = X_train.size();
    const size_t NUM_FEATURES = X_train[0].size(); // K = 4 features
    cout << "\nSo mau train (N): " << NUM_SAMPLES // 70 
         << ", So dac trung (K): " << NUM_FEATURES // 8
         << ", So nhan (L): " << NUM_LABELS << endl; // 3 

    // 5. Mã hóa 
    print_example_banner("5. Ma hoa du lieu");
    // Kích thước của Batch (số lượng mẫu có thể đóng gói)
    size_t slot_count = encoder.slot_count();
    cout << "So luong mau co the dong goi Slot count: " << slot_count << endl; // 4096

    cout << "-- Ma hoa du lieu train" << endl;
    // 5.1 Mã hóa C_W_col kich thước 1x1 (mỗi plaintext kích thước Nx1)
    cout << "5.1 Ma hoa vector trong so W_train..." << endl;
    vector<double> W_train(NUM_SAMPLES, 1.0);
    Ciphertext C_W_col;
    Plaintext ptw;
    encoder.encode(W_train, scale, ptw); 
    encryptor.encrypt(ptw, C_W_col);

    // 5.2. Mã hóa C_X_col kích thước Kx1 (mỗi plaintext kích thước Nx1) , Mã hóa C_Y_col kích thước Lx1 (mỗi plaintext kích thước Nx1)
    cout << "5.2 Ma hoa cac dac trung X_train va Y_train..." << endl;
    vector<vector<double>> X_train_T(NUM_FEATURES, vector<double>(NUM_SAMPLES, 0.0)); // ma trận KxN
    vector<vector<double>> Y_train_T(NUM_LABELS, vector<double>(NUM_SAMPLES, 0.0)); // ma trận LxN
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        for (size_t j = 0; j < NUM_FEATURES; ++j) {
            X_train_T[j][i] = X_train[i][j]; // Đặc trưng j, mẫu i
        }
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            Y_train_T[l][i] = Y_train_onehot[i][l]; // Nhãn l, mẫu i
        }
    }

    vector<Ciphertext> C_X_cols(NUM_FEATURES); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_FEATURES; ++j) {
        // cout << "Ma hoa feature cho dac trung X thu " << j << "..." << endl;
        Plaintext ptx;
        encoder.encode(X_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols[j]);
    }

    vector<Ciphertext> C_Y_col(NUM_LABELS); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_LABELS; ++j) {
        // cout << "Ma hoa feature cho dac trung Y thu " << j << "..." << endl;
        Plaintext ptx;
        encoder.encode(Y_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_Y_col[j]);
    }
    
    //???
    cout << "-- Ma hoa du lieu test" << endl;
    vector<vector<double>> X_test, Y_test_onehot;
    extract_data(split.test, X_test, Y_test_onehot); // X_test là ma trận NxK (70x7) ,Y_test_onehot là ma trận N x L (70x3)

    const size_t NUM_SAMPLES_TEST = X_test.size();
    const size_t NUM_FEATURES_TEST = X_test[0].size(); // K = 4 features
    cout << "\nSo mau test (N): " << NUM_SAMPLES_TEST // 
            << ", So dac trung (K): " << NUM_FEATURES_TEST //
            << ", So nhan (L): " << NUM_LABELS << endl; // 

    cout << "5.1 Ma hoa vector trong so W_train_test..." << endl;
    vector<double> W_train_test (NUM_SAMPLES_TEST, 1.0);
    Ciphertext C_W_col_test ;
    Plaintext ptw_test ;
    encoder.encode(W_train_test , scale, ptw_test ); 
    encryptor.encrypt(ptw_test , C_W_col_test );

    cout << "5.2 Ma hoa cac dac trung X_test va Y_test..." << endl;
    vector<vector<double>> X_test_T(NUM_FEATURES_TEST, vector<double>(NUM_SAMPLES_TEST, 0.0)); // ma trận KxN
    vector<vector<double>> Y_test_T(NUM_LABELS, vector<double>(NUM_SAMPLES_TEST, 0.0)); // ma trận LxN
    for (size_t i = 0; i < NUM_SAMPLES_TEST; ++i) {
        for (size_t j = 0; j < NUM_FEATURES_TEST; ++j) {
            X_test_T[j][i] = X_test[i][j]; // Đặc trưng j, mẫu i
        }
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            Y_test_T[l][i] = Y_test_onehot[i][l]; // Nhãn l, mẫu i
        }
    }

    vector<Ciphertext> C_X_cols_test(NUM_FEATURES_TEST); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_FEATURES_TEST; ++j) {
        Plaintext ptx;
        encoder.encode(X_test_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols_test[j]);
    }

    vector<Ciphertext> C_Y_col_test(NUM_LABELS); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_LABELS; ++j) {
        Plaintext ptx;
        encoder.encode(Y_test_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_Y_col_test[j]);
    }

    cout << "-- Ma hoa nguong." << endl;
    set<double> unique_thresholds_set;
    for (const auto& row : X_train) {
        for (double val : row) {
            unique_thresholds_set.insert(val);
        }
    }
    vector<double> all_thresholds(unique_thresholds_set.begin(), unique_thresholds_set.end());
    all_thresholds.resize(number_threshold); // Giảm số ngưỡng để thử nghiệm nhanh

    print_example_banner("6. Tree training");  
    unique_ptr<Node_c> root = train_decision_tree(
        C_X_cols, C_W_col, C_Y_col, all_thresholds, 0, max_depth,
        evaluator, encryptor, decryptor, encoder, relin_keys, gal_keys, context, scale
    );
    
    cout << "Huan luyen cay quyet dinh hoan tat voi do sau : " << max_depth << "" << endl;

    // luu model cay vao file
    save_model(root, src_model_tree);

    // Tải
    unique_ptr<Node_c> root2 = load_model(src_model_tree , context);

    print_example_banner("7. Du doan Bao mat (Prediction)");

    // Trọng số ban đầu cho dự đoán là C_W_col (tất cả là 1.0)
    vector<Ciphertext> C_Final_Prediction = predict_decision_tree(
        root, C_X_cols_test, C_W_col_test, evaluator, encryptor, encoder, 
        relin_keys, context, scale, gal_keys
    );
    cout << "-- Giai ma ket qua du doan..." << endl;
    vector<vector<double>> decoded_predictions; // Kích thước NUM_SAMPLES_TEST * NUM_LABELS
    for (size_t l = 0; l < NUM_LABELS; ++l) {
        Plaintext pt_final_pred;
        decryptor.decrypt(C_Final_Prediction[l], pt_final_pred);
        vector<double> decoded_col;
        encoder.decode(pt_final_pred, decoded_col);
        decoded_predictions.push_back(decoded_col);
    }
    cout << "Ket qua du doan cho N mau (Lx1 vector trong moi slot):" << endl;
    cout << "Mau | Nhan Setosa (0) | Nhan Versicolor (1) | Nhan Virginica (2) | Nhan Du Doan Cuoi" << endl;
    cout << "----|-----------------|---------------------|--------------------|-------------------" << endl;

    for (size_t i = 0; i < NUM_SAMPLES_TEST; ++i) {
        double score_setosa = decoded_predictions[0][i]; // Lấy giá trị cột đầu tiên
        double score_versicolor = decoded_predictions[1][i]; // Lấy giá trị cột tiếp theo
        double score_virginica = decoded_predictions[2][i]; // Lấy giá trị cột thứ 3
        // Tạo vector để tìm max
        vector<double> scores = {score_setosa, score_versicolor, score_virginica};
        
        // Tìm điểm số lớn nhất
        double max_score = scores[0];
        size_t final_label = 0;

        for (size_t j = 1; j < NUM_LABELS; ++j) {
            if (scores[j] > max_score) {
                max_score = scores[j];
                final_label = j;
            }
        }
        
        // In kết quả chi tiết
        cout << setw(3) << i << " | " 
            << setw(15) << fixed << setprecision(5) << score_setosa << " | " 
            << setw(19) << fixed << setprecision(5) << score_versicolor << " | " 
            << setw(18) << fixed << setprecision(5) << score_virginica << " | "
            << setw(16) << final_label << " (" << max_score << ")" << endl;
    }

    double acc = calculate_accuracy(decoded_predictions, Y_test_onehot);
    cout << "Accuracy = " << acc * 100.0 << "%\n";

    double f1_macro = calculate_f1_score(decoded_predictions, Y_test_onehot);
    cout << "F1-Macro Score = " << f1_macro << "\n";

    return 0;
}
