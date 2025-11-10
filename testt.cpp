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

int max_example = 5; // Số lượng mẫu Iris để đọc từ file CSV
int NUM_SAMPLES = max_example + 1;
int NUM_FEATURES = 4; // Số đặc trưng trong tập dữ liệu iris
int NUM_LABELS = 3; // Iris-setosa, Iris-versicolor, Iris-virginica
int max_depth = 2; 

size_t MAX_DEGREE = 6; // soft_step_evaluation() Giới hạn bậc đa thức xấp xỉ đến z^4 vì độ phức tạp tính toán
// soft_step_evaluation(hệ số bậc 16 đã dùng code python tính toán)
// const vector<double> SOFT_STEP_COEFFICIENTS_16 = {
//     // c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16
//      5.00000000e-01,  2.11445799e+00,  1.15591931e-10, -6.38009501e+00,
//     5.00000000e-01,  2.11445799e+00,  1.15591931e-10, -6.38009501e+00,
//    -4.91650318e-10,  1.09534390e+01,  8.95471329e-10, -1.01295272e+01,
//    -8.38669100e-10,  5.27558906e+00,  4.36355800e-10, -1.54908047e-01, // Sửa: Giảm 1 bậc mũ của 1.54908047e+00 để phù hợp với kết quả
//    -1.27390646e-10,  2.39094701e-01,  1.95169389e-11, -1.50730122e-02,
//    -1.22081599e-12
// };
const vector<double> SOFT_STEP_COEFFICIENTS_16 ={5.00000000e-01,  9.72913227e-01, -3.43964120e-16, -4.65617910e-01,  7.72516483e-17,  7.53639755e-02};

struct Sample {
    vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
    string label;            // "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

vector<Sample> read_iris_csv(const string &filename) {
    cout << "read_iris_csv() read data from " << filename << endl;
    vector<Sample> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Khong the mo file: " + filename);
    }
    int m = 0;
    int check = 150/max_example - 1;
    string line;
    while (getline(file, line)) {
        if (m % check == 0){
            stringstream ss(line);
            string item;
            Sample iris;
            for (int i = 0; i < NUM_FEATURES; ++i) {
                if (!getline(ss, item, ',')) {
                    throw runtime_error("Dinh dang du lieu khong hop le trong file.");
                }
                iris.features.push_back(stod(item));
            }
            if (!getline(ss, item, ',')) {
                throw runtime_error("Dinh dang du lieu khong hop le trong file.");
            }
            iris.label = item;
            data.push_back(iris);
        }
        m++;
    }
    file.close();
    cout << "So mau doc duoc: " << data.size() << endl;
    return data;
}

void normalize(vector<Sample> &data)
{
    for (int j = 0; j < NUM_FEATURES; j++) {
        double min_val = 1e9, max_val = -1e9;

        for (auto &row : data) {
            min_val = min(min_val, row.features[j]);
            max_val = max(max_val, row.features[j]);
        }

        if (abs(max_val - min_val) < 1e-12) continue;

        for (auto &row : data) {
            row.features[j] = 2 * (row.features[j] - min_val) / (max_val - min_val) - 1;
        }
    }
}

// Hàm lấy thông tin của 1 ciphertext 
void print_ct_info(const SEALContext &context, const Ciphertext &ct, const std::string &name) {
    auto context_data = context.get_context_data(ct.parms_id());
    size_t chain_index = context_data->chain_index();
    cout << name 
         << ": chain_index=" << chain_index 
         << ", scale=" << ct.scale() 
         << ", pol_mod_deg=" << context_data->parms().poly_modulus_degree()
         << " parms_id=" << ct.parms_id() 
         << " size=" << ct.size() 
         << " (poly_count per component)" << endl;
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

Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    double scale,
    const SEALContext &context,
    Decryptor &decryptor
) ///?
{
    cout << "soft_step_evaluation()" << endl;
    // 1. Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(MAX_DEGREE + 1); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // Tính toán các Lũy thừa của z 
    // đảm bảo encrypted_z=cx - theta đã ở đúng level và scale ban đầu mong muốn, i < SOFT_STEP_COEFFICIENTS_16.size() là bão hòa nhưng máy cấu hình kém 
    for (size_t i = 1; i < MAX_DEGREE; ++i) {
        // cout << " Computing power z^" << (i+1) << endl;
        Ciphertext z_for_mul = encrypted_z;
        if (z_for_mul.parms_id() != powers.back().parms_id()) {
            evaluator.mod_switch_to_inplace(z_for_mul, powers.back().parms_id());
        }

        double s1 = powers.back().scale();
        double s2 = z_for_mul.scale();
        // cout << "  scales before mul: power=" << s1 << ", z_copy=" << s2 << endl;

        Ciphertext tmp;
        try {
            evaluator.multiply(powers.back(), z_for_mul, tmp); // tmp scale ≈ s1 * s2
        } catch (const exception &e) {
            cerr << "MULTIPLY EXCEPTION at i=" << i << ": " << e.what() << endl;
            throw;
        }

        // Relinearize và rescale như trước
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.rescale_to_next_inplace(tmp);

        // print_ct_info(context, tmp, ("tmp after multiply and rescale z^" + to_string(i+1)).c_str());

        powers.push_back(tmp);
    }

    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], scale, c0_plain);

    Ciphertext result;
    encryptor.encrypt(c0_plain, result); // dùng Encryptor, không phải Evaluator

    evaluator.mod_switch_to_inplace(result, powers.back().parms_id());

    // 2. Tính tổng đa thức: c0 + c1*[[z^1]] + c2*[[z^2]] + ... + c16*[[z^16]]
    for (size_t i = 1; i < MAX_DEGREE; ++i) { // i < SOFT_STEP_COEFFICIENTS_16.size()
        // cout << " Adding term for z^" << i << endl;

        // Đảm bảo i không vượt quá số lượng hệ số thực tế (nếu SOFT_STEP_COEFFICIENTS_16 nhỏ hơn 5)
        if (i >= SOFT_STEP_COEFFICIENTS_16.size()) break;
        double coeff = SOFT_STEP_COEFFICIENTS_16[i];
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
    Decryptor &decryptor)
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
        C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, decryptor); 
    Ciphertext C_Phi_Left = soft_step_evaluation(
        C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, decryptor); 

    // Tính W_phi = W * Phi
    // cout << "Tinh W_Phi" << endl;
    Ciphertext C_W_Phi_Right, C_W_Phi_Left;
    
    // Right: C_W_col * C_Phi_Right
    // cout << " Right: C_W_col * C_Phi_Right" << endl;
    align_and_multiply(C_W_col, C_Phi_Right, C_W_Phi_Right);
    vector<double> values_1 = decrypt_to_vector(C_W_Phi_Right, decryptor, encoder);
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
                    cout << "1" << endl;
                    Ciphertext t1;
                    align_and_multiply(C_W_col, C_Y_cols[l], t1);      
                    align_and_multiply(t1, C_Phi_Right, C_Term_Right_tmp); 
                } else if (attempt == "(W*Phi)*Y") {
                    cout << "2" << endl;
                    align_and_multiply(C_W_Phi_Right, C_Y_cols[l], C_Term_Right_tmp);
                } else { // (Phi*Y)*W
                    cout << "3" << endl;
                    Ciphertext t1;
                    align_and_multiply(C_Phi_Right, C_Y_cols[l], t1);
                    align_and_multiply(t1, C_W_col, C_Term_Right_tmp);
                }
                C_right_counts[l] = C_Term_Right_tmp;
                done = true;
                vector<double> values_1 = decrypt_to_vector(C_right_counts[l], decryptor, encoder);
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

// compute_gini_impurity() Tính độ tạp Gini (Gini Impurity) cho một phân chia cụ thể
double compute_gini_impurity(
    const vector<double>& right_counts, // Vector Lx1 cleartext (tổng trọng số bên phải)
    const vector<double>& left_counts)  // Vector Lx1 cleartext (tổng trọng số bên trái)
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


// Hàm phụ trợ để tính W_phi cho ngưỡng đã chọn
pair<Ciphertext, Ciphertext> compute_W_phi_best(
    int best_feature, double best_threshold,
    const vector<Ciphertext>& C_X_cols, const Ciphertext& C_W_col, 
    Evaluator& evaluator, Encryptor& encryptor, const CKKSEncoder& encoder,
    const RelinKeys& relin_keys, double scale, const SEALContext& context,
    Decryptor &decryptor
)
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
    Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context, decryptor); 
    Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, decryptor); 

    // W_new = W * Phi
    Ciphertext C_W_new_right, C_W_new_left;
    
    // Right: W * Phi_Right
    align_and_multiply(C_W_col, C_Phi_Right, C_W_new_right);
    align_and_multiply(C_W_col, C_Phi_Left, C_W_new_left);
    
    cout << "Completed compute_W_phi_best()" << endl;
    return {C_W_new_right, C_W_new_left};
}

int main(){
    print_example_banner("0. Xu ly du lieu");
    //  Đọc dữ liệu Iris từ file CSV
    vector<Sample> iris_data;
    try {
        iris_data = read_iris_csv("C:/hu/decision-tree-he/Project_Build/Release/iris.csv");
        cout << "Du lieu sau khi doc tu file:" << endl;
        for (const auto &iris : iris_data) {
            cout << "Features: ";
            for (const auto &feature : iris.features) {
                cout << feature << " ";
            }
            cout << "| Label: " << iris.label << endl;
        }
    } catch (const exception &e) {
        cerr << "Loi: " << e.what() << endl;
        return 1;
    }

    // Chuẩn hóa về [-1, 1] để phù hợp với HE
    normalize(iris_data);
    cout << "\nDu lieu sau khi chuan hoa ve [-1, 1]:" << endl;
    for (const auto &iris : iris_data) {
        cout << "Features: ";
        for (const auto &feature : iris.features) {
            cout << feature << " ";
        }
        cout << "| Label: " << iris.label << endl;
    }

    // Chuyển nhãn sang one-hot encoding
    map<string, int> label_map = {
        {"setosa", 0},
        {"versicolor", 1},
        {"virginica", 2}
    };
    auto extract_data = [&](const vector<Sample>& iris_data, vector<vector<double>>& X, vector<vector<double>>& Y_onehot) {
        if (iris_data.empty()) return;
        
        int iii = 0;
        for (const auto& iris : iris_data) {
            X.push_back(iris.features);
            vector<double> y_onehot(NUM_LABELS, 0.0); 
            if (label_map.count(iris.label)) {
                int label_index = label_map.at(iris.label);
                y_onehot[label_index] = 1.0;
            } else {
                cerr << "Canh bao: Nhãn khong hop le: " << iris.label << endl;
            }
            Y_onehot.push_back(y_onehot);
            iii++;
        }
    };
    vector<vector<double>> X_train, Y_train_onehot;
    extract_data(iris_data, X_train, Y_train_onehot);
    cout << "\nDu lieu sau khi chuyen sang one-hot encoding:" << endl;
    for (size_t i = 0; i < X_train.size(); ++i) {
        cout << "Sample " << i + 1 << " -> Features: ";
        for (const auto &f : X_train[i]) {
            cout << f << " ";
        }

        cout << "| One-hot Label: ";
        for (const auto &y : Y_train_onehot[i]) {
            cout << y << " ";
        }
        cout << endl;
    }
    
    // Tạo context SEAL và các thành phần cần thiết khác ở đây...
    print_example_banner("1. Thiet lap tham so CKKS");
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = pow(2, 14); // Tăng bậc đa thức để thêm chỗ cho moduli
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 60}
    ));
    double scale = pow(2.0, 40);
    SEALContext context(parms);
    cout << endl;
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, keygen.secret_key());
    CKKSEncoder encoder(context);

    print_example_banner("3. Ma hoa du lieu (C_W_col, X_train_T, Y_train_T)");
    size_t slot_count = encoder.slot_count();
    cout << "So luong mau co the dong goi, Slot count: " << slot_count << endl; // 8192 , = poly_modulus_degree/2 vì biểu diễn 1 số phức =  2 só thực

    // 5.1 Mã hóa C_W_col kich thước 1x1 (mỗi plaintext kích thước Nx1)
    vector<double> W_train(NUM_SAMPLES, 1.0);
    Ciphertext C_W_col;
    Plaintext ptw;
    encoder.encode(W_train, scale, ptw); 
    encryptor.encrypt(ptw, C_W_col);
    print_ct_info(context, C_W_col, "C_W_col ");

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

    cout << "Gia tri X_train sau khi chuyen vi" << endl;
    for (size_t i = 0; i < X_train_T.size(); ++i) {
        cout << "Features " << i + 1 << " | ";
        for (const auto &f : X_train_T[i]) {
            cout << f << " ";
        }
        cout << endl;
    }

    cout << "Gia tri One-hot sau khi chuyen vi" << endl;
    for (size_t i = 0; i < Y_train_T.size(); ++i) {
        cout << "One-hot Label " << i + 1 << " | ";
        for (const auto &y : Y_train_T[i]) {
            cout << y << " ";
        }
        cout << endl;
    }

    vector<Ciphertext> C_X_cols(NUM_FEATURES); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_FEATURES; ++j) {
        Plaintext ptx;
        encoder.encode(X_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols[j]);
    }

    vector<Ciphertext> C_Y_cols(NUM_LABELS); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_LABELS; ++j) {
        Plaintext ptx;
        encoder.encode(Y_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_Y_cols[j]);
    }

    print_example_banner("4. Tap hop cac nguong duy nhat tu tap du lieu");   
    double all_thresholds = {0.5};

    // double z = 1.3; 
    // Plaintext plain_z;
    // encoder.encode(z, scale, plain_z);
    // Ciphertext C_Z_left;
    // encryptor.encrypt(plain_z, C_Z_left);
    // Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context, decryptor); 

    // vector<double> values = decrypt_to_vector(C_Phi_Left, decryptor, encoder);
    // cout << "Values: ";
    // int i;
    // for (i = 0; i < 5; i++) {
    //     cout << values[i] << " ";
    // }
    // cout << endl;

    auto [C_right_counts, C_left_counts] = compute_weighted_counts_homo(
        0, all_thresholds, C_X_cols, C_W_col, C_Y_cols, 
        evaluator, encryptor, encoder, relin_keys, galois_keys, context, scale, decryptor
    );
    Ciphertext C_right_counts_1 = C_right_counts[0];
    Ciphertext C_left_counts_1 = C_left_counts[0];
    vector<double> values = decrypt_to_vector(C_right_counts_1, decryptor, encoder);
    cout << "C_right_counts[0]: ";
    int i;
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;
    values = decrypt_to_vector(C_right_counts[1], decryptor, encoder);
    cout << "C_right_counts[1]: ";
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;
    values = decrypt_to_vector(C_right_counts[2], decryptor, encoder);
    cout << "C_right_counts[2]: ";
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;
    values = decrypt_to_vector(C_left_counts[0], decryptor, encoder);
    cout << "C_left_counts[0]: ";
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;
    values = decrypt_to_vector(C_left_counts[1], decryptor, encoder);
    cout << "C_left_counts[1]: ";
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;
    values = decrypt_to_vector(C_left_counts[2], decryptor, encoder);
    cout << "C_left_counts[2]: ";
    for (i = 0; i < 3; i++) {
        cout << values[i] << " ";
    }
    cout << endl;

    // auto [C_W_new_right, C_W_new_left] = compute_W_phi_best(
    //     0, 0.5, C_X_cols, C_W_col, 
    //     evaluator, encryptor, encoder, relin_keys, scale, context, decryptor
    // );
    // vector<double> values = decrypt_to_vector(C_W_new_right, decryptor, encoder);
    // cout << "C_W_new_right: ";
    // int i;
    // for (i = 0; i < 6; i++) {
    //     cout << values[i] << " ";
    // }
    // cout << endl;

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
    
    double current_gini = compute_gini_impurity(right_counts_clear, left_counts_clear);
    cout << "current_gini : " << current_gini << endl;

    return 0;
}