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

int number_threshold = 2; // Số ngưỡng để thử nghiệm trong train_decision_tree() thực tế số ngưỡng N x k = 70 x 8 = 560 
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

    file.close();
    return data;
}

void normalize(vector<Sample> &data)
{
    if (data.empty() || data[0].features.empty()) return;
    size_t NUM_SAMPLES = data[0].features.size();
    for (int j = 0; j < NUM_SAMPLES; j++) {
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

SplitData split_data(vector<Sample> data, double train_ratio=0.8) {
    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    size_t n = data.size();
    int n_train = static_cast<int>(train_ratio * n);

    SplitData split;
    split.train = vector<Sample>(data.begin(), data.begin() + n_train);
    split.test  = vector<Sample>(data.begin() + n_train, data.end());
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
vector<double> decrypt_to_vector(
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

// leaf_value(W, Y) = sum(W * Y)
Ciphertext leaf_value(
    const Ciphertext &C_W_col,         
    const Ciphertext &C_Y_col,        
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const GaloisKeys &galois_keys,
    const SEALContext &context,
    int NUM_SAMPLES
)
{
    cout << "leaf_value()" << endl;

    Ciphertext W = C_W_col;
    Ciphertext Y = C_Y_col;

    print_ct_info(context, W, "\tbefore align C_W_col");
    print_ct_info(context, Y, "\tbefore align C_Y_col");

    auto w_ci = context.get_context_data(W.parms_id())->chain_index();
    auto y_ci = context.get_context_data(Y.parms_id())->chain_index();
    if (w_ci > y_ci) {
        evaluator.mod_switch_to_inplace(W, Y.parms_id());
        cout << "mod-switched W down to Y's level" << endl;
    } else if (y_ci > w_ci) {
        evaluator.mod_switch_to_inplace(Y, W.parms_id());
        cout << "mod-switched Y down to W's level" << endl;
    }

    // print_ct_info(context, W, "\tafter align C_W_col");
    // print_ct_info(context, Y, "\tafter align C_Y_col");

    // size_t common_ci = context.get_context_data(W.parms_id())->chain_index();
    // if (common_ci < 1) {
    //     cerr << "leaf_value ERROR: insufficient levels (common chain_index < 1). Cannot multiply+rescale safely." << endl;
    //     throw runtime_error("leaf_value: insufficient levels for multiply");
    // }

    Ciphertext P;
    try {
        evaluator.multiply(W, Y, P);                 // <-- use W and Y, not C_W_col/C_Y_col
        evaluator.relinearize_inplace(P, relin_keys);
        evaluator.rescale_to_next_inplace(P);
    } catch (const exception &e) {
        cerr << "leaf_value MULTIPLY/RESCALE exception: " << e.what() << endl;
        print_ct_info(context, W, "\tW (at failure)");
        print_ct_info(context, Y, "\tY (at failure)");
        throw;
    }

    Ciphertext res = P;
    for (int step = 1; step < NUM_SAMPLES; step+= 1) {
        Ciphertext tmp;
        evaluator.rotate_vector(res, step, galois_keys, tmp);
        evaluator.add_inplace(res, tmp);
    }

    cout << "Completed leaf_value() = " << endl;
    return res;
}

// 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
// 2. Tính tổng đa thức: c0 + c1*[[z^1]] + c2*[[z^2]] + ... + c16*[[z^16]]
Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    double scale,
    const SEALContext &context,
    vector<double> SOFT_STEP_COEFFICIENTS
) ///?
{
    cout << "soft_step_evaluation()" << endl;
    // 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(SOFT_STEP_COEFFICIENTS.size()); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // Tính toán các Lũy thừa của z 
    // đảm bảo encrypted_z=cx - theta đã ở đúng level và scale ban đầu mong muốn, i < SOFT_STEP_COEFFICIENTS_16.size() là bão hòa nhưng máy cấu hình kém 
    for (size_t i = 1; i < SOFT_STEP_COEFFICIENTS.size(); ++i) {
        // cout << " Computing power z^" << (i+1) << endl;
        Ciphertext tmp;
        Ciphertext z_for_mul = encrypted_z;

        if(i == 1 || i%2 == 0){ // i + 1 == 2 || (i+1)%2 != 0
            if (z_for_mul.parms_id() != powers.back().parms_id()) {
                evaluator.mod_switch_to_inplace(z_for_mul, powers.back().parms_id());
            }
            try {
                evaluator.multiply(powers.back(), z_for_mul, tmp); // tmp scale ≈ s1 * s2
            } catch (const exception &e) {
                cerr << "MULTIPLY EXCEPTION at i=" << i << ": " << e.what() << endl;
                throw;
            }
        }else{
            try {
                int z = (i+1)/2 - 1;
                Ciphertext& ciphertext_i = powers[(z)]; 
                evaluator.multiply(ciphertext_i, ciphertext_i, tmp); // tmp scale ≈ s1 * s2
            } catch (const exception &e) {
                cerr << "MULTIPLY EXCEPTION at i=" << i << ": " << e.what() << endl;
                throw;
            }
        }

        // double s1 = powers.back().scale();
        // double s2 = z_for_mul.scale();
        // cout << "  scales before mul: power=" << s1 << ", z_copy=" << s2 << endl;

        // Ciphertext tmp;
        // try {
        //     evaluator.multiply(powers.back(), z_for_mul, tmp); // tmp scale ≈ s1 * s2
        // } catch (const exception &e) {
        //     cerr << "MULTIPLY EXCEPTION at i=" << i << ": " << e.what() << endl;
        //     throw;
        // }

        // Relinearize và rescale như trước
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.rescale_to_next_inplace(tmp);

        // print_ct_info(context, tmp, ("tmp after multiply and rescale z^" + to_string(i+1)).c_str());

        powers.push_back(tmp);
    }

    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS[0], scale, c0_plain);

    Ciphertext result;
    encryptor.encrypt(c0_plain, result); // dùng Encryptor, không phải Evaluator

    evaluator.mod_switch_to_inplace(result, powers.back().parms_id());

    // 2. Tính tổng đa thức: c0 + c1*[[z^1]] + c2*[[z^2]] + ... + c16*[[z^16]]
    for (size_t i = 1; i < SOFT_STEP_COEFFICIENTS.size(); ++i) { // i < SOFT_STEP_COEFFICIENTS_16.size()
        // cout << " Adding term for z^" << i << endl;

        // Đảm bảo i không vượt quá số lượng hệ số thực tế (nếu SOFT_STEP_COEFFICIENTS_16 nhỏ hơn 5)
        if (i >= SOFT_STEP_COEFFICIENTS.size()) break;
        
        double coeff = SOFT_STEP_COEFFICIENTS[i];
        if (std::abs(coeff) < 1e-12) continue;

        // powers[i-1] là z^i
        Ciphertext &power = powers[i-1];

        // 2.1 Encode hệ số với scale phù hợp (scale cần giống với power.scale())
        double power_scale = power.scale(); 
        Plaintext coeff_plain;
        encoder.encode(coeff, power_scale, coeff_plain);

        // 2.2 Nếu coeff_plain.parms_id() khác với power.parms_id(), mod-switch coeff_plain
        evaluator.mod_switch_to_inplace(coeff_plain, power.parms_id());

        // 2.3 Nhân: term = coeff * z^i
        Ciphertext term;
        evaluator.multiply_plain(power, coeff_plain, term);
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

// compute_weighted_counts_homo(W, Y, X) = sum(W, soft_step_evaluation(cx[i]-theta), cyx)
pair<vector<Ciphertext>, vector<Ciphertext>> compute_weighted_counts_homo(
    int feature_idx, // 1 số plaintext 
    double threshold, // 1 số plaintext 
    const vector<Ciphertext>& C_X_cols, // K ciphertext
    const Ciphertext& C_W_col,          // 1 ciphertextMULTIPLY EXCEPTION
    const vector<Ciphertext>& C_Y_cols,   // L ciphertext
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& galois_keys,
    const SEALContext& context,
    double scale, // tính soft_step_evaluation()
    int NUM_SAMPLES
)
{
    cout << "compute_weighted_counts_homo()" << endl;
        
    // Helper: align levels then multiply (safe)
    auto align_and_multiply = [&](const Ciphertext &A, const Ciphertext &B, Ciphertext &Out) {
        // Make copies so originals are not mutated
        Ciphertext a = A;
        Ciphertext b = B;

        auto a_ci = context.get_context_data(a.parms_id())->chain_index();
        auto b_ci = context.get_context_data(b.parms_id())->chain_index();
        // cout << "  align_and_multiply: a.chain_index=" << a_ci << ", b.chain_index=" << b_ci << endl;

        // If levels differ, mod-switch the one on higher level down to the other's level
        if (a.parms_id() != b.parms_id()) {
            if (a_ci > b_ci) {
                evaluator.mod_switch_to_inplace(a, b.parms_id());
                // cout << "   mod-switched a down to b's level" << endl;
            } else {
                evaluator.mod_switch_to_inplace(b, a.parms_id());
                // cout << "   mod-switched b down to a's level" << endl;
            }
        }

        // After alignment, get the (common) chain index
        auto common_ci = context.get_context_data(a.parms_id())->chain_index();
        // cout << "   common chain_index=" << common_ci << endl;

        // Need at least chain_index >= 1 to allow rescale_to_next after multiply
        if (common_ci < 1) {
            cerr << "   ERROR: insufficient levels (chain_index < 1). Cannot multiply-and-rescale." << endl;
            throw runtime_error("Insufficient levels for multiply");
        }

        // cout << "   scales before mul: a.scale=" << a.scale() << ", b.scale=" << b.scale() << endl;

        try {
            evaluator.multiply(a, b, Out);
            evaluator.relinearize_inplace(Out, relin_keys);
            evaluator.rescale_to_next_inplace(Out);
        } catch (const exception &e) {
            cerr << "   MULTIPLY EXCEPTION: " << e.what() << endl;
            print_ct_info(context, a, "   a (after align)");
            print_ct_info(context, b, "   b (after align)");
            throw;
        }
    };

    // cout << "compute_weighted_counts_homo()" << endl;
    const size_t NUM_LABELS = C_Y_cols.size(); // 3 
    // cout << "NUM_LABELS=" << NUM_LABELS << endl; //3 
    // Khởi tạo vector kết quả Lx1 (một ciphertext cho mỗi nhãn)
    vector<Ciphertext> C_right_counts(NUM_LABELS);
    vector<Ciphertext> C_left_counts(NUM_LABELS);
    
    // --- BƯỚC 1: TÍNH SOFT-STEP VÀ TRỌNG SỐ TẠI SLOT (W_Phi) ---
    
    // Mã hóa Ngưỡng Theta (theta)
    Plaintext pt_theta;
    encoder.encode(threshold, C_X_cols[feature_idx].scale(), pt_theta);
    Ciphertext C_Theta;
    encryptor.encrypt(pt_theta, C_Theta);

    // Tính Độ lệch Z_right = X[i] - theta
    Ciphertext C_Z_right; 
    evaluator.sub(C_X_cols[feature_idx], C_Theta, C_Z_right); 

    // Tính Độ lệch Z_left = theta - X[i]
    Ciphertext C_Z_left; 
    evaluator.sub(C_Theta, C_X_cols[feature_idx], C_Z_left); 

    // cout << "Tinh Soft-Step" << endl;
    // Tính Soft-Step: Phi(Z_right) và Phi(Z_left)
    // *Lưu ý: Phải đảm bảo soft_step_evaluation đã được cập nhật đúng tham số*
    Ciphertext C_Phi_Right = soft_step_evaluation(
        C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); 
    Ciphertext C_Phi_Left = soft_step_evaluation(
        C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); 

    // Tính W_phi = W * Phi
    // cout << "Tinh W_Phi" << endl;
    Ciphertext C_W_Phi_Right, C_W_Phi_Left;
    
    // Right: C_W_col * C_Phi_Right
    // cout << " Right: C_W_col * C_Phi_Right" << endl;
    align_and_multiply(C_W_col, C_Phi_Right, C_W_Phi_Right);
    // cout << "Right: C_W_col * C_Phi_Right: ";
    // int i;
    // for (i = 0; i < 8; i++) {
    //     cout << values_1[i] << "    ";
    // }
    // cout << endl;

    // Left: C_W_col * C_Phi_Left
    // cout << " Left: C_W_col * C_Phi_Left" << endl;
    align_and_multiply(C_W_col, C_Phi_Left, C_W_Phi_Left);
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
                    align_and_multiply(C_W_col, C_Y_cols[l], t1);      
                    align_and_multiply(t1, C_Phi_Right, C_Term_Right_tmp); 
                } else if (attempt == "(W*Phi)*Y") {
                    align_and_multiply(C_W_Phi_Right, C_Y_cols[l], C_Term_Right_tmp);
                } else { // (Phi*Y)*W
                    Ciphertext t1;
                    align_and_multiply(C_Phi_Right, C_Y_cols[l], t1);
                    align_and_multiply(t1, C_W_col, C_Term_Right_tmp);
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
        for (int step = 1; step < NUM_SAMPLES ; step += 1) {
            Ciphertext C_Rotated;
            evaluator.rotate_vector(C_Sum_Right_l, step, galois_keys, C_Rotated);
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
                    align_and_multiply(C_W_col, C_Y_cols[l], t1);
                    align_and_multiply(t1, C_Phi_Left, C_Term_Left_tmp);
                } else if (attempt == "(W*Phi)*Y") {
                    align_and_multiply(C_W_Phi_Left, C_Y_cols[l], C_Term_Left_tmp);
                } else { // (Phi*Y)*W
                    Ciphertext t1;
                    align_and_multiply(C_Phi_Left, C_Y_cols[l], t1);
                    align_and_multiply(t1, C_W_col, C_Term_Left_tmp);
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
            evaluator.rotate_vector(C_Sum_Left_l, step, galois_keys, C_Rotated);
            evaluator.add_inplace(C_Sum_Left_l, C_Rotated);
        }
        C_left_counts[l] = C_Sum_Left_l; // tong bang so phan tu dau tien
    }

    cout << "Completed compute_weighted_counts_homo()" << endl;
    return {C_right_counts, C_left_counts}; // vecto<cipher=vecto Nx1>
}

struct Node {
    bool is_leaf = false;
    // Giá trị nút lá (Nếu là lá, kích thước Lx1)
    vector<double> leaf_value; 

    // Thông số phân chia (Nếu không là lá)
    int feature_index = -1;
    double threshold = 0.0; 

    // Các nhánh (Node con)
    unique_ptr<Node> left_child = nullptr;
    unique_ptr<Node> right_child = nullptr;
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
    int best_feature, double best_threshold,
    const vector<Ciphertext>& C_X_cols, const Ciphertext& C_W_col, 
    Evaluator& evaluator, Encryptor& encryptor, const CKKSEncoder& encoder,
    const RelinKeys& relin_keys, double scale, const SEALContext& context)
{
    cout << "compute_W_phi_best()" << endl;
    auto align_and_multiply = [&](const Ciphertext& a, const Ciphertext& b, Ciphertext& result) {
        Ciphertext a_copy = a;
        Ciphertext b_copy = b;

        auto a_level = context.get_context_data(a_copy.parms_id())->chain_index();
        auto b_level = context.get_context_data(b_copy.parms_id())->chain_index();

        if (a_level > b_level) {
            evaluator.mod_switch_to_inplace(a_copy, b_copy.parms_id());
        } else if (b_level > a_level) {
            evaluator.mod_switch_to_inplace(b_copy, a_copy.parms_id());
        }

        evaluator.multiply(a_copy, b_copy, result);
        evaluator.relinearize_inplace(result, relin_keys);
        evaluator.rescale_to_next_inplace(result);
    };

    // Mã hóa Ngưỡng Theta
    Plaintext pt_theta;
    encoder.encode(best_threshold, C_X_cols[best_feature].scale(), pt_theta);
    Ciphertext C_Theta;
    encryptor.encrypt(pt_theta, C_Theta);

    // Độ lệch Z
    cout << " 1" << endl;
    Ciphertext C_Z_right, C_Z_left; 
    evaluator.sub(C_X_cols[best_feature], C_Theta, C_Z_right); 
    evaluator.sub(C_Theta, C_X_cols[best_feature], C_Z_left); 

    // Soft-Step
    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); 
    Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_16); 
    
    // --- DEBUG: skip soft_step_evaluation, use constant Phi ---
    // cout << " 2" << endl;
    // Plaintext pt_phi_right, pt_phi_left;
    // encoder.encode(0.5, scale, pt_phi_right); // or 1.0, tuỳ bạn
    // encoder.encode(0.5, scale, pt_phi_left);
    // cout << " 3" << endl;
    // Ciphertext C_Phi_Right, C_Phi_Left;
    // encryptor.encrypt(pt_phi_right, C_Phi_Right);
    // encryptor.encrypt(pt_phi_left, C_Phi_Left);

    // W_new = W * Phi
    Ciphertext C_W_new_right, C_W_new_left;
    
    // Right: W * Phi_Right
    align_and_multiply(C_W_col, C_Phi_Right, C_W_new_right);
    align_and_multiply(C_W_col, C_Phi_Left, C_W_new_left);
    
    cout << "Completed compute_W_phi_best()" << endl;
    return {C_W_new_right, C_W_new_left};
}

unique_ptr<Node> train_decision_tree(
    const vector<Ciphertext>& C_X_cols,
    const Ciphertext& C_W_col, 
    const vector<Ciphertext>& C_Y_cols,
    const vector<double>& all_thresholds, // Tập hợp các ngưỡng duy nhất
    const vector<double>& W_train_clear, // Trọng số cleartext (cần để phân tách cho đệ quy)
    int depth,
    int max_depth,
    Evaluator& evaluator,
    Encryptor& encryptor,
    Decryptor& decryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& galois_keys,
    const SEALContext& context,
    double scale,
    size_t NUM_SAMPLES
){
    cout << "train_decision_tree() depth=" << depth << endl;
    auto node = make_unique<Node>();
    const size_t NUM_FEATURES = C_X_cols.size();
    size_t NUM_LABELS = C_Y_cols.size();

    // 1. ĐIỀU KIỆN DỪNG
    if (depth >= max_depth) {
        // cout << " Reached max depth. Creating leaf node." << endl;
        node->is_leaf = true;
        node->leaf_value.resize(NUM_LABELS); // Khởi tạo vector kết quả Lx1

        // Lặp qua TẤT CẢ L nhãn để tính tổng trọng số cho từng nhãn
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            // cout << "  Computing leaf value for label " << l << endl;
            // 1.1. TÍNH TỔNG TRỌNG SỐ ĐỒNG HÌNH (C_W * C_Y[l])
            Ciphertext C_weighted_sum = leaf_value(
                C_W_col, C_Y_cols[l], evaluator, relin_keys, galois_keys, context, NUM_SAMPLES 
            );

            // 1.2. GIẢI MÃ (Client-side)
            Plaintext pt_result;
            decryptor.decrypt(C_weighted_sum, pt_result);
            
            vector<double> decoded_result;
            encoder.decode(pt_result, decoded_result);
            
            // 1.3. LƯU GIÁ TRỊ LÁ (Slot 0 chứa tổng cuối cùng)
            node->leaf_value[l] = decoded_result[0];
        }
        
        // cout << "Created leaf node at depth=" << depth << " with values: ";
        return node; 
    }

    // 2. TÌM NGƯỠNG TỐI ƯU
    double min_gini = 1e9;
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // (Vector này sẽ lưu các giá trị W*phi(Z) cho lần gọi đệ quy sau)
    Ciphertext C_W_Phi_Best_Right, C_W_Phi_Best_Left;

    // Lặp qua TẤT CẢ thuộc tính (i) và TẤT CẢ ngưỡng (theta)
    for (int i = 0; i < NUM_FEATURES; ++i) { // i < NUM_FEATURES
        // cout << " Feature " << i << endl;
        for (double threshold : all_thresholds) { // 163 ngưỡng // double threshold : all_thresholds
            // cout << "  Threshold " << threshold << endl;
            // 2.1. Tính tổng trọng số bảo mật (Homomorphic Counts)
            auto [C_right_counts, C_left_counts] = compute_weighted_counts_homo(
                i, threshold, C_X_cols, C_W_col, C_Y_cols, 
                evaluator, encryptor, encoder, relin_keys, galois_keys, context, scale, NUM_SAMPLES
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
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // --- BƯỚC CỐT LÕI: TÍNH W_NEW CHO ĐỆ QUY ---
    
    // 3.1. Tính các tập trọng số mã hóa mới (W_new_right và W_new_left)
    auto [C_W_new_right, C_W_new_left] = compute_W_phi_best(
        best_feature, best_threshold, C_X_cols, C_W_col, 
        evaluator, encryptor, encoder, relin_keys, scale, context
    );

    // 4. GỌI ĐỆ QUY (RECURSION)
    
    // Nút Con Phải (Sử dụng W_new_right làm trọng số đầu vào)
    node->right_child = train_decision_tree(
        C_X_cols, C_W_new_right, C_Y_cols, all_thresholds, W_train_clear, 
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, galois_keys, context, scale, NUM_SAMPLES
    );

    // Nút Con Trái (Sử dụng W_new_left làm trọng số đầu vào)
    node->left_child = train_decision_tree(
        C_X_cols, C_W_new_left, C_Y_cols, all_thresholds, W_train_clear,
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, galois_keys, context, scale, NUM_SAMPLES
    );

    cout << "Completed train_decision_tree() at depth=" << depth << endl;
    return node;
}

Ciphertext predict_decision_tree(
    const unique_ptr<Node>& node,       // Nút hiện tại của cây
    const vector<Ciphertext>& C_X_cols, // K ciphertext (Dữ liệu đầu vào, Column Batched)
    const Ciphertext& C_W_current,      // Trọng số hiện tại của mẫu (W * phi * phi...)
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const SEALContext& context,
    double scale)
{
    cout << "predict_decision_tree()" << endl;

    if (!node) {
        // cout << " Node is null, returning zero ciphertext." << endl;
        Ciphertext zero_ct = C_W_current;
        evaluator.negate(zero_ct, zero_ct);
        Plaintext pt_zero;
        encoder.encode(0.0, C_W_current.scale(), pt_zero);
        evaluator.add_plain_inplace(zero_ct, pt_zero);
        return zero_ct;
    }

    // NÚT LÁ
    if (node->is_leaf) {
        // cout << " Reached leaf node. Processing leaf values." << endl;
        const size_t NUM_LABELS = node->leaf_value.size();

        Plaintext pt_leaf;
        encoder.encode(node->leaf_value, C_W_current.scale(), pt_leaf);
        evaluator.mod_switch_to_inplace(pt_leaf, C_W_current.parms_id());

        Ciphertext C_Leaf_Output;
        evaluator.multiply_plain(C_W_current, pt_leaf, C_Leaf_Output);
        return C_Leaf_Output;
    }

    // KHÔNG PHẢI LÁ — TÍNH SOFT-STEP
    int i_best = node->feature_index;
    double theta_best = node->threshold;

    const Ciphertext& C_X_i = C_X_cols[i_best];

    Plaintext pt_theta;
    encoder.encode(theta_best, C_X_i.scale(), pt_theta);
    Ciphertext C_Theta;
    encryptor.encrypt(pt_theta, C_Theta);
    evaluator.mod_switch_to_inplace(C_Theta, C_X_i.parms_id());

    Ciphertext C_Z_right, C_Z_left;
    evaluator.sub(C_X_i, C_Theta, C_Z_right);
    evaluator.sub(C_Theta, C_X_i, C_Z_left);

    // Soft-step (ở đây tạm hardcode = 0.5 để test)
    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_8);
    Ciphertext C_Phi_Left  = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, SOFT_STEP_COEFFICIENTS_8);


    // NHÂN TRỌNG SỐ MỚI (W_new)
    auto safe_multiply = [&](const Ciphertext& C_W, const Ciphertext& C_Phi, Ciphertext& C_Out, string tag)
    {
        Ciphertext W = C_W;
        Ciphertext Phi = C_Phi;

        auto w_data = context.get_context_data(W.parms_id());
        auto p_data = context.get_context_data(Phi.parms_id());
        size_t w_level = w_data->chain_index();
        size_t p_level = p_data->chain_index();

        // Align levels
        if (w_level > p_level)
            evaluator.mod_switch_to_inplace(W, Phi.parms_id());
        else if (p_level > w_level)
            evaluator.mod_switch_to_inplace(Phi, W.parms_id());

        // Align scales
        if (fabs(W.scale() - Phi.scale()) > 1.0) {
            double new_scale = min(W.scale(), Phi.scale());
            W.scale() = new_scale;
            Phi.scale() = new_scale;
        }

        try {
            evaluator.multiply(W, Phi, C_Out);
            evaluator.relinearize_inplace(C_Out, relin_keys);
            evaluator.rescale_to_next_inplace(C_Out);
        } catch (const std::exception &e) {
            cerr << " Multiply failed (" << tag << "): " << e.what() << endl;
            cerr << "  W level=" << w_level << ", Phi level=" << p_level << endl;
            cerr << "  W scale=" << W.scale() << ", Phi scale=" << Phi.scale() << endl;
            throw;
        }
    };

    Ciphertext C_W_Right, C_W_Left;
    // cout << " Multiplying W_current * Phi_Right..." << endl;
    safe_multiply(C_W_current, C_Phi_Right, C_W_Right, "Right");

    // cout << " Multiplying W_current * Phi_Left..." << endl;
    safe_multiply(C_W_current, C_Phi_Left, C_W_Left, "Left");

    // GỌI ĐỆ QUY
    Ciphertext C_Output_Right = predict_decision_tree(
        node->right_child, C_X_cols, C_W_Right, evaluator, encryptor, encoder, 
        relin_keys, context, scale
    );

    Ciphertext C_Output_Left = predict_decision_tree(
        node->left_child, C_X_cols, C_W_Left, evaluator, encryptor, encoder, 
        relin_keys, context, scale
    );

    // CỘNG DỒN KẾT QUẢ
    if (C_Output_Right.parms_id() != C_Output_Left.parms_id()) {
        auto ci_r = context.get_context_data(C_Output_Right.parms_id())->chain_index();
        auto ci_l = context.get_context_data(C_Output_Left.parms_id())->chain_index();
        if (ci_r > ci_l)
            evaluator.mod_switch_to_inplace(C_Output_Right, C_Output_Left.parms_id());
        else
            evaluator.mod_switch_to_inplace(C_Output_Left, C_Output_Right.parms_id());
    }

    Ciphertext C_Final_Output;
    evaluator.add(C_Output_Right, C_Output_Left, C_Final_Output);

    cout << "Completed predict_decision_tree()" << endl;
    return C_Final_Output;
}

void calculate_accuracy(
    const std::vector<double>& decoded_predictions, 
    const std::vector<std::vector<double>>& Y_test_onehot,
    size_t NUM_SAMPLES,
    size_t num_labels) 
{
    cout << "calculate_accuracy()" << endl;
    
    if (decoded_predictions.size() < NUM_SAMPLES * num_labels) {
        std::cerr << "Loi: Kich thuoc decoded_predictions khong du." << std::endl;
        return;
    }

    int correct_predictions = 0;

    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        // 1. Xác định Nhãn Dự đoán (Predicted Label)
        std::vector<double> sample_scores;
        for (size_t j = 0; j < num_labels; ++j) {
            // Lấy score của nhãn j cho mẫu i. Giả định dữ liệu giải mã liền kề.
            sample_scores.push_back(decoded_predictions[i * num_labels + j]);
        }

        // Tìm chỉ mục (index) của điểm số lớn nhất (Predicted Label)
        auto max_it = std::max_element(sample_scores.begin(), sample_scores.end());
        int predicted_label = std::distance(sample_scores.begin(), max_it);

        // 2. Xác định Nhãn Thực tế (True Label)
        // Tìm chỉ mục (index) của giá trị 1.0 trong vector one-hot
        size_t true_label = -1;
        for (size_t j = 0; j < num_labels; ++j) {
            // Sử dụng một ngưỡng nhỏ để tính nhãn (vì có thể có lỗi làm tròn)
            if (std::abs(Y_test_onehot[i][j] - 1.0) < 1e-6) { 
                true_label = j;
                break;
            }
        }

        // 3. So sánh
        if (predicted_label == true_label) {
            correct_predictions++;
        }
        
        // Optional: In chi tiết từng mẫu
        /*
        std::cout << "Mau " << i << ": True=" << true_label 
                  << ", Predicted=" << predicted_label 
                  << (predicted_label == true_label ? " (CORRECT)" : " (INCORRECT)") 
                  << std::endl;
        */
    }

    // 4. Tính và Hiển thị Accuracy
    double accuracy = (double)correct_predictions / NUM_SAMPLES;
    std::cout << "\nTong so mau: " << NUM_SAMPLES << std::endl;
    std::cout << "Du doan dung: " << correct_predictions << std::endl;
    std::cout << "**Accuracy: " << std::fixed << std::setprecision(4) << accuracy << "**" << std::endl;

    cout << "Completed calculate_accuracy()" << endl;
}

int main()
{
    // ====== 1. Thiết lập tham số CKKS ======
    print_example_banner("1. Thiet lap tham so CKKS");
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = pow(2, 15); 
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60} // 20 tầng
    ));

    double scale = pow(2.0, 40);
    SEALContext context(parms);

    print_parameters(context);
    cout << endl;

    // ====== 2. Sinh khóa ======
    print_example_banner("2. Sinh khoa");
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    // auto galois_keys = keygen.create_galois_keys(); 
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, keygen.secret_key());
    CKKSEncoder encoder(context);

    // ====== 3. Đọc dữ liệu Iris ======
    print_example_banner("3. Doc du lieu Iris");
    vector<Sample> data;
    try {
        data = read_csv_dynamic("C:/hu/decision-tree-he/Project_Build/Release/iris.csv");
        if (data.empty()) throw runtime_error("Khong doc duoc du lieu iris.");
        normalize(data);
    } catch (const exception &e) {
        cerr << "Loi: " << e.what() << endl;
        return 1;
    }

    // 4. Xử lý dữ liệu 
    print_example_banner("4. Xu ly du lieu");
    SplitData split = split_data(data);

    // one-hot encoding: "Iris-setosa" -> 0, "Iris-versicolor" -> 1, "Iris-virginica" -> 2
    map<string, int> label_map = {
        {"setosa", 0},
        {"versicolor", 1},
        {"virginica", 2}
    };
    const size_t NUM_LABELS = label_map.size(); 
    cout << "So nhan (L) NUM_LABELS = " << NUM_LABELS << endl; // L = 3
    
    // Trích xuất dữ liệu từ SplitData
    auto extract_data = [&](const vector<Sample>& iris_data, vector<vector<double>>& X, vector<vector<double>>& Y_onehot) {
        if (iris_data.empty()) return;
        size_t NUM_SAMPLES = iris_data[0].features.size();
        
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

    // 5.1 Mã hóa C_W_col kich thước 1x1 (mỗi plaintext kích thước Nx1)
    vector<double> W_train(NUM_SAMPLES, 1.0);
    Ciphertext C_W_col;
    Plaintext ptw;
    encoder.encode(W_train, scale, ptw); 
    encryptor.encrypt(ptw, C_W_col);

    // 5.2. Mã hóa C_X_col kích thước Kx1 (mỗi plaintext kích thước Nx1) , Mã hóa C_Y_col kích thước Lx1 (mỗi plaintext kích thước Nx1)
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
    
    // 5.3 Mã hóa C_Y kích thước Nx1 (mỗi plaintext ở bản rõ kích thước Lx1)
    // vector<Ciphertext> C_Y(Y_train_onehot.size()); // Kích thước ĐÚNG là N (số mẫu = 70) 
    // for (size_t j = 0; j < Y_train_onehot.size() ; ++j) {
    //     // cout << "Ma hoa nhan cho dac trung Y thu " << j+1 << "..." << endl;
    //     Plaintext pty;
    //     encoder.encode(Y_train_onehot[j], scale, pty); 
    //     encryptor.encrypt(pty, C_Y[j]);
    // }
    
    //???
    vector<vector<double>> X_test, Y_test_onehot;
    extract_data(split.test, X_test, Y_test_onehot); // X_test là ma trận NxK (70x7) ,Y_test_onehot là ma trận N x L (70x3)

    const size_t NUM_SAMPLES_TEST = X_test.size();
    const size_t NUM_FEATURES_TEST = X_test[0].size(); // K = 4 features
    cout << "\nSo mau test (N): " << NUM_SAMPLES_TEST // 
            << ", So dac trung (K): " << NUM_FEATURES_TEST //
            << ", So nhan (L): " << NUM_LABELS << endl; // 

    vector<double> W_train_test (NUM_SAMPLES_TEST, 1.0);
    Ciphertext C_W_col_test ;
    Plaintext ptw_test ;
    encoder.encode(W_train_test , scale, ptw_test ); 
    encryptor.encrypt(ptw_test , C_W_col_test );

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

    //???

    // 6. Tree training và các bước tiếp theo...
    print_example_banner("6. Tree training va cac buoc tiep theo");   

    // Tập hợp các ngưỡng duy nhất từ dữ liệu huấn luyện
    set<double> unique_thresholds_set;
    for (const auto& row : X_train) {
        for (double val : row) {
            unique_thresholds_set.insert(val);
        }
    }
    vector<double> all_thresholds(unique_thresholds_set.begin(), unique_thresholds_set.end());

    all_thresholds.resize(number_threshold); // Giảm số ngưỡng để thử nghiệm nhanh

    int max_depth = 2; 
    unique_ptr<Node> root = train_decision_tree(
        C_X_cols, C_W_col, C_Y_col, all_thresholds, W_train, 0, max_depth,
        evaluator, encryptor, decryptor, encoder, relin_keys, galois_keys, context, scale, NUM_SAMPLES
    );
    
    cout << "Huan luyen cay quyet dinh hoan tat voi do sau : " << max_depth << "" << endl;

    // ====== 7. DỰ ĐOÁN BẢO MẬT (PREDICTION) ======
    print_example_banner("7. Du doan Bao mat (Prediction)");

    // Trọng số ban đầu cho dự đoán là C_W_col (tất cả là 1.0)
    
    Ciphertext C_Final_Prediction = predict_decision_tree(
        root, C_X_cols_test, C_W_col_test, evaluator, encryptor, encoder, 
        relin_keys, context, scale
    );

    // 7.1. GIẢI MÃ KẾT QUẢ CUỐI CÙNG
    Plaintext pt_final_pred;
    decryptor.decrypt(C_Final_Prediction, pt_final_pred);
    
    vector<double> decoded_predictions;
    encoder.decode(pt_final_pred, decoded_predictions);

    // 7.2. HIỂN THỊ KẾT QUẢ DỰ ĐOÁN (L=3 nhãn)
    // 7.2. HIỂN THỊ KẾT QUẢ DỰ ĐOÁN (L=3 nhãn)
    cout << "Ket qua du doan cho N mau (Lx1 vector trong moi slot):" << endl;
    cout << "Mau | Nhan Setosa (0) | Nhan Versicolor (1) | Nhan Virginica (2) | Nhan Du Doan Cuoi" << endl;
    cout << "----|-----------------|---------------------|--------------------|-------------------" << endl;

    // L = 3 nhãn (số lượng lớp)
    const size_t L_NUM_LABELS = 3; 

    for (size_t i = 0; i < NUM_SAMPLES_TEST; ++i) {
        
        // Giả định: Kết quả cho mẫu i nằm ở các vị trí i, i+1, i+2 trong decoded_predictions
        // (Đây là giả định dựa trên cách in của bạn, phù hợp với Column Batching)
        
        double score_setosa = decoded_predictions[i];
        double score_versicolor = decoded_predictions[i + 1];
        double score_virginica = decoded_predictions[i + 2];

        // Tạo vector để tìm max
        vector<double> scores = {score_setosa, score_versicolor, score_virginica};
        
        // Tìm điểm số lớn nhất
        double max_score = scores[0];
        size_t final_label = 0;

        for (size_t j = 1; j < L_NUM_LABELS; ++j) {
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

    calculate_accuracy(
        decoded_predictions, 
        Y_test_onehot, 
        NUM_SAMPLES_TEST, 
        L_NUM_LABELS
    );

    return 0;
}
