#include "examples.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <set>

using namespace std;
using namespace seal;

struct Iris {
    vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
    string label;            // "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

vector<Iris> read_iris_csv(const string &filename) {
    // in ra "read_iris_csv" 
    cout << "read_iris_csv() Doc du lieu tu file: " << filename << endl;
    vector<Iris> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Khong the mo file: " + filename);
    }
    int m = 0;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        Iris iris;
        for (int i = 0; i < 8; ++i) {
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
        // cout << "Read iris: " << iris.label << endl;
        m++;
        if (m == 5) break; // Chỉ đọc 5 dòng đầu tiên để kiểm tra
    }
    file.close();
    return data;
}

void normalize(vector<Iris> &data)
{
    if (data.empty() || data[0].features.empty()) return;

    size_t num_features = data[0].features.size();

    for (int j = 0; j < num_features; j++) {
        double min_val = 1e9, max_val = -1e9;

        // Tìm giá trị min/max cho mỗi cột
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
    vector<Iris> train, val, test;
};

SplitData split_data(vector<Iris> data, double train_ratio=0.7, double val_ratio=0.15) {
    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    size_t n = data.size();
    int n_train = static_cast<int>(train_ratio * n);
    int n_val = static_cast<int>(val_ratio * n);

    SplitData split;
    split.train = vector<Iris>(data.begin(), data.begin() + n_train);
    split.val   = vector<Iris>(data.begin() + n_train, data.begin() + n_train + n_val);
    split.test  = vector<Iris>(data.begin() + n_train + n_val, data.end());
    return split;
}

// leaf_value(W, Y) = sum(W * Y)
Ciphertext leaf_value(
    const Ciphertext &C_W_col,         // ciphertext chứa W per slot
    const Ciphertext &C_Y_col,         // ciphertext chứa Y per slot
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const GaloisKeys &galois_keys,
    const SEALContext &context)
{
    cout << "leaf_value()" << endl;

    // Make editable copies (we will mod-switch copies, not originals)
    Ciphertext W = C_W_col;
    Ciphertext Y = C_Y_col;

    // Debug helper
    auto print_info = [&](const string &name, const Ciphertext &ct) {
        auto ctx_data = context.get_context_data(ct.parms_id());
        size_t ci = ctx_data ? ctx_data->chain_index() : -1;
        cout << name << ": chain_index=" << ci << ", scale=" << ct.scale() << endl;
    };

    print_info("before align W", W);
    print_info("before align Y", Y);

    // Align levels: mod-switch higher down to lower
    auto w_ci = context.get_context_data(W.parms_id())->chain_index();
    auto y_ci = context.get_context_data(Y.parms_id())->chain_index();
    if (w_ci > y_ci) {
        evaluator.mod_switch_to_inplace(W, Y.parms_id());
        cout << "mod-switched W down to Y's level" << endl;
    } else if (y_ci > w_ci) {
        evaluator.mod_switch_to_inplace(Y, W.parms_id());
        cout << "mod-switched Y down to W's level" << endl;
    }

    print_info("after align W", W);
    print_info("after align Y", Y);

    // Re-fetch common chain_index and ensure we have at least one level to rescale after multiply.
    size_t common_ci = context.get_context_data(W.parms_id())->chain_index();
    if (common_ci < 1) {
        cerr << "leaf_value ERROR: insufficient levels (common chain_index < 1). Cannot multiply+rescale safely." << endl;
        throw runtime_error("leaf_value: insufficient levels for multiply");
    }

    // Now multiply the ALIGNED copies (IMPORTANT: use W and Y, not originals)
    Ciphertext P;
    try {
        cout << "1: multiplying aligned W and Y..." << endl;
        evaluator.multiply(W, Y, P);                 // <-- use W and Y, not C_W_col/C_Y_col
        cout << "2: relinearize..." << endl;
        evaluator.relinearize_inplace(P, relin_keys);
        cout << "3: rescale..." << endl;
        evaluator.rescale_to_next_inplace(P);
    } catch (const exception &e) {
        cerr << "leaf_value MULTIPLY/RESCALE exception: " << e.what() << endl;
        print_info("W (at failure)", W);
        print_info("Y (at failure)", Y);
        throw;
    }

    // Sum across slots by rotations (use slot_count/2 as safe upper bound)
    // If you have encoder here you can use encoder.slot_count()/2; otherwise keep conservative value.
    int max_offset = 1 << 10; // 1024 conservative
    Ciphertext res = P;
    for (int step = 1; step <= max_offset; step *= 2) {
        Ciphertext tmp;
        evaluator.rotate_vector(res, step, galois_keys, tmp);
        evaluator.add_inplace(res, tmp);
    }

    cout << "Completed leaf_value()" << endl;
    return res;
}

// Hàm lấy thông tin của 1 ciphertext 
void print_ct_info(const SEALContext &context, const Ciphertext &ct, const std::string &name) {
    auto context_data = context.get_context_data(ct.parms_id());
    size_t chain_index = context_data->chain_index();
    cout << name << ": chain_index=" << chain_index 
         << ", scale=" << ct.scale() << ", pol_mod_deg=" << context_data->parms().poly_modulus_degree()
         << endl;
}

// soft_step(hệ số bậc 16 đã dùng code python tính toán)
const vector<double> SOFT_STEP_COEFFICIENTS_16 = {
    // c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16
    5.00000000e-01,  2.11445799e+00,  1.15591931e-10, -6.38009501e+00,
   -4.91650318e-10,  1.09534390e+01,  8.95471329e-10, -1.01295272e+01,
   -8.38669100e-10,  5.27558906e+00,  4.36355800e-10, -1.54908047e-01, // Sửa: Giảm 1 bậc mũ của 1.54908047e+00 để phù hợp với kết quả
   -1.27390646e-10,  2.39094701e-01,  1.95169389e-11, -1.50730122e-02,
   -1.22081599e-12
};

Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    double scale,
    const SEALContext &context)
{
    cout << "soft_step_evaluation()" << endl;
    const size_t MAX_DEGREE = 3; // Giới hạn bậc đa thức xấp xỉ đến z^4 vì độ phức tạp tính toán
    // 1) Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(MAX_DEGREE + 1); // powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // IMPORTANT: đảm bảo encrypted_z đã ở đúng level và scale ban đầu mong muốn, i < SOFT_STEP_COEFFICIENTS_16.size() là bão hòa nhưng máy cấu hình kém 
    for (size_t i = 1; i < MAX_DEGREE; ++i) {
        // cout << " Computing power z^" << (i+1) << endl;

        // Tạo bản sao của encrypted_z và mod-switch nó về level của powers.back()
        Ciphertext z_for_mul = encrypted_z;
        if (z_for_mul.parms_id() != powers.back().parms_id()) {
            evaluator.mod_switch_to_inplace(z_for_mul, powers.back().parms_id());
        }

        // (Tùy chọn) kiểm tra scale tương đương (chỉ log, không cố gắng thay đổi)
        double s1 = powers.back().scale();
        double s2 = z_for_mul.scale();
        // cout << "  scales before mul: power=" << s1 << ", z_copy=" << s2 << endl;

        // Multiply (giờ cả 2 cùng parms_id)
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

    // 2) Mã hoá hệ số tự do c0 và khởi tạo result
    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], scale, c0_plain);

    Ciphertext result;
    encryptor.encrypt(c0_plain, result); // dùng Encryptor, không phải Evaluator

    // Mod-switch result xuống level cuối cùng của các lũy thừa (powers[MAX_DEGREE - 1])
    evaluator.mod_switch_to_inplace(result, powers.back().parms_id());

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
        // kết quả term có cùng parms_id với power; scale của term là power.scale() * power_scale (≈ scale^2)
        evaluator.rescale_to_next_inplace(term);   // rescale để đưa scale về ~scale

        // 2.4 Điều chỉnh levels/parms_id: trước khi cộng, result và term phải cùng parms_id
        // target là parms_id của result (đã mod-switched trước vòng lặp)
        auto target_parms_id = result.parms_id();
        if (term.parms_id() != target_parms_id) {
            evaluator.mod_switch_to_inplace(term, target_parms_id);
        }
        // cộng term vào result
        term.scale() = result.scale();
        evaluator.add_inplace(result, term);
    }
    cout << "Completed soft_step_evaluation()" << endl;
    return result;
}

// split(W, Y, X) = sum(W, soft_step_evaluation(cx[i]-theta), cyx)
pair<vector<Ciphertext>, vector<Ciphertext>> compute_weighted_counts_homo(
    int feature_idx,
    double threshold,
    const vector<Ciphertext>& C_X_cols, // K ciphertext
    const Ciphertext& C_W_col,          // 1 ciphertext
    const vector<Ciphertext>& C_Y_cols,   // L ciphertext
    Evaluator& evaluator,
    Encryptor& encryptor,
    const CKKSEncoder& encoder,
    const RelinKeys& relin_keys,
    const GaloisKeys& galois_keys,
    const SEALContext& context,
    double scale,
    size_t num_samples)
{
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
            print_ct_info(context, Out, "   Out AFTER mul+relin+rescale");
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
    // Ciphertext C_Phi_Right = soft_step_evaluation(
    //     C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context); 
    // Ciphertext C_Phi_Left = soft_step_evaluation(
    //     C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context); 

    // --- DEBUG: skip soft_step_evaluation, use constant Phi ---
    Plaintext pt_phi_right, pt_phi_left;
    encoder.encode(0.5, scale, pt_phi_right); // or 1.0, tuỳ bạn
    encoder.encode(0.5, scale, pt_phi_left);

    Ciphertext C_Phi_Right, C_Phi_Left;
    encryptor.encrypt(pt_phi_right, C_Phi_Right);
    encryptor.encrypt(pt_phi_left, C_Phi_Left);

    // Tính W_phi = W * Phi
    // cout << "Tinh W_Phi" << endl;
    Ciphertext C_W_Phi_Right, C_W_Phi_Left;
    
    // Right: C_W_col * C_Phi_Right
    // cout << " Right: C_W_col * C_Phi_Right" << endl;
    align_and_multiply(C_W_col, C_Phi_Right, C_W_Phi_Right);

    // Left: C_W_col * C_Phi_Left
    // cout << " Left: C_W_col * C_Phi_Left" << endl;
    align_and_multiply(C_W_col, C_Phi_Left, C_W_Phi_Left);
    
    // --- BƯỚC 2: TÍNH TỔNG THEO TỪNG NHÃN L (Summation) ---
    // cout << "Tinh Tong theo tung nhan L" << endl;
    for (size_t l = 0; l < NUM_LABELS; ++l) {
        auto print_ct_debug = [&](const string &name, const Ciphertext &ct) {
            auto ctx_data = context.get_context_data(ct.parms_id());
            size_t ci = ctx_data ? ctx_data->chain_index() : -1;
            // cout << name << ": chain_index=" << ci 
            //     << ", scale=" << setprecision(12) << ct.scale()
            //     << ", parms_id? " << (ct.parms_id().data() ? "yes":"no") << endl;
        };

        // Safe multiply helper (align levels, multiply, relin, rescale)
        auto safe_mul = [&](Ciphertext a, Ciphertext b, Ciphertext &out) {
            // align levels by mod-switching the higher down to lower
            auto a_ci = context.get_context_data(a.parms_id())->chain_index();
            auto b_ci = context.get_context_data(b.parms_id())->chain_index();
            if (a.parms_id() != b.parms_id()) {
                if (a_ci > b_ci) evaluator.mod_switch_to_inplace(a, b.parms_id());
                else evaluator.mod_switch_to_inplace(b, a.parms_id());
            }
            auto common_ci = context.get_context_data(a.parms_id())->chain_index();
            if (common_ci < 1) {
                // cannot multiply+rescale safely
                throw runtime_error("safe_mul: insufficient levels (chain_index < 1)");
            }
            // print_ct_debug("safe_mul - aligned a", a);
            // print_ct_debug("safe_mul - aligned b", b);

            evaluator.multiply(a, b, out);
            evaluator.relinearize_inplace(out, relin_keys);
            evaluator.rescale_to_next_inplace(out);
            // print_ct_debug("safe_mul - out(after mul+rescale)", out);
        };

        // ======================= Diagnostic prints BEFORE the failing multiply =======================
        // Right after you computed C_W_Phi_Right and C_W_Phi_Left (they exist here)
        print_ct_debug("DBG C_W_Phi_Right", C_W_Phi_Right);
        print_ct_debug("DBG C_W_Phi_Left", C_W_Phi_Left);

        // Print one Y column (we will iterate, but start with l=0)
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            print_ct_debug("DBG C_Y_cols[" + to_string(l) + "]", C_Y_cols[l]);
        }

        // ======================= Try multiple multiply orders for each label =======================
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            // cout << "Processing label " << l << " (TRY best multiply order)" << endl;

            // Print current state
            // print_ct_debug("  C_W_Phi_Right", C_W_Phi_Right);
            // print_ct_debug("  C_Y_cols[l]", C_Y_cols[l]);

            Ciphertext C_Term_Right_tmp;

            // Strategy: try (W*Y) then *Phi  OR (W*Phi) then *Y  OR (Phi*Y) then *W
            // We attempt an order that leaves at least one level for subsequent multiply.
            bool done = false;
            vector<string> attempts = {"(W*Y)*Phi", "(W*Phi)*Y", "(Phi*Y)*W"};
            for (auto &attempt : attempts) {
                try {
                    // cout << " Attempting order: " << attempt << endl;
                    if (attempt == "(W*Y)*Phi") {
                        // compute t1 = W * Y
                        Ciphertext t1;
                        safe_mul(C_W_col, C_Y_cols[l], t1);         // consumes one level -> chain_index--
                        // then align t1 with Phi and multiply
                        safe_mul(t1, C_Phi_Right, C_Term_Right_tmp); 
                    } else if (attempt == "(W*Phi)*Y") {
                        // this is your current approach: we already have C_W_Phi_Right as W*Phi
                        // but check chain_index of C_W_Phi_Right
                        // print_ct_debug("   pre-check C_W_Phi_Right", C_W_Phi_Right);
                        safe_mul(C_W_Phi_Right, C_Y_cols[l], C_Term_Right_tmp);
                    } else { // (Phi*Y)*W
                        Ciphertext t1;
                        safe_mul(C_Phi_Right, C_Y_cols[l], t1);
                        safe_mul(t1, C_W_col, C_Term_Right_tmp);
                    }
                    // if reached here no exception => success
                    // cout << "  Success with order: " << attempt << endl;
                    // store and mark done
                    C_right_counts[l] = C_Term_Right_tmp;
                    done = true;
                    break;
                } catch (const exception &e) {
                    cout << "  Attempt " << attempt << " failed: " << e.what() << endl;
                    // continue to next attempt
                }
            } // end attempts

            if (!done) {
                // None of the orders worked: we are out of levels.
                cerr << "ERROR: cannot compute W*Phi*Y for label " << l 
                    << " with current levels. Consider increasing coeff_modulus or reducing operations." << endl;
                // throw or handle gracefully; here we throw to stop and show debug
                throw runtime_error("compute_weighted_counts_homo: insufficient levels for W*Phi*Y");
            }

            // Then do rotations/sums on C_right_counts[l] as before (if done)
            Ciphertext C_Sum_Right_l = C_right_counts[l];
            for (int step = 1; step <= int(num_samples / 2); step *= 2) {
                Ciphertext C_Rotated;
                evaluator.rotate_vector(C_Sum_Right_l, step, galois_keys, C_Rotated);
                evaluator.add_inplace(C_Sum_Right_l, C_Rotated);
            }
            C_right_counts[l] = C_Sum_Right_l;
        }

        // --- LEFT SIDE (Tương tự) ---
        // Tương tự cho C_W_Phi_Left
        // --- LEFT SIDE (Giống RIGHT, có thử nhiều thứ tự multiply) ---
        bool done_left = false;
        Ciphertext C_Term_Left_tmp;
        vector<string> left_attempts = {"(W*Y)*Phi", "(W*Phi)*Y", "(Phi*Y)*W"};

        for (auto &attempt : left_attempts) {
            try {
                if (attempt == "(W*Y)*Phi") {
                    Ciphertext t1;
                    safe_mul(C_W_col, C_Y_cols[l], t1);
                    safe_mul(t1, C_Phi_Left, C_Term_Left_tmp);
                } else if (attempt == "(W*Phi)*Y") {
                    safe_mul(C_W_Phi_Left, C_Y_cols[l], C_Term_Left_tmp);
                } else { // (Phi*Y)*W
                    Ciphertext t1;
                    safe_mul(C_Phi_Left, C_Y_cols[l], t1);
                    safe_mul(t1, C_W_col, C_Term_Left_tmp);
                }

                C_left_counts[l] = C_Term_Left_tmp;
                done_left = true;
                break;
            } catch (const exception &e) {
                cout << "  LEFT attempt " << attempt << " failed: " << e.what() << endl;
            }
        }

        if (!done_left) {
            cerr << "ERROR: cannot compute LEFT W*Phi*Y for label " << l
                << " with current levels. Consider increasing coeff_modulus or reducing operations." << endl;
            throw runtime_error("compute_weighted_counts_homo: insufficient levels for LEFT W*Phi*Y");
        }

        // --- SUM ROTATIONS for LEFT (giống RIGHT) ---
        Ciphertext C_Sum_Left_l = C_left_counts[l];
        for (int step = 1; step <= int(num_samples / 2); step *= 2) {
            Ciphertext C_Rotated;
            evaluator.rotate_vector(C_Sum_Left_l, step, galois_keys, C_Rotated);
            evaluator.add_inplace(C_Sum_Left_l, C_Rotated);
        }
        C_left_counts[l] = C_Sum_Left_l;
    }

    cout << "Completed compute_weighted_counts_homo()" << endl;
    return {C_right_counts, C_left_counts};
}

// total_side() = sum(side[i, theta][l]) với side={left, right}, side[i, theta][l] là số mẫu ở nút con bên trái/phải có nhãn l

// gini_weight() = sum_side(1-sum_l((side[i, theta][l])/total_side)^2).total_side[i, theta] với side={left, right}

// gini_theta_best() = min_theta(gini_weight())

// train_decision_tree(X, Y, W, depth, v={v.leaf_value hoặc v.right/v.leaf})
// if depth >= max_depth thì return leaf_value(W, Y)
// else
// each i, theta: compute gini_theta_best()
// cập nhật v.feature, v.threshold, wx_right, wx_left
// return train_decision_tree(X, Y, W_left, depth+1, v_left) và train_decision_tree(X, Y, W_right, depth+1, v_right)

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

// Tính độ tạp Gini (Gini Impurity) cho một phân chia cụ thể
double compute_gini_impurity(
    const vector<double>& right_counts, // Vector Lx1 cleartext (tổng trọng số bên phải)
    const vector<double>& left_counts)  // Vector Lx1 cleartext (tổng trọng số bên trái)
{
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
        gini_right = (1.0 - sum_sq) * (1 / total_all);
    }

    double gini_left = 0.0;
    if (total_left > 1e-9) {
        double sum_sq = 0.0;
        for (double count : left_counts) {
            double prob = count / total_left;
            sum_sq += prob * prob;
        }
        gini_left = (1.0 - sum_sq) * (1 / total_all);
    }

    return gini_right + gini_left;
}

// Hàm phụ trợ để tính W_phi cho ngưỡng đã chọn
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
    // Ciphertext C_Phi_Right = soft_step_evaluation(C_Z_right, evaluator, encryptor, encoder, relin_keys, scale, context); 
    // Ciphertext C_Phi_Left = soft_step_evaluation(C_Z_left, evaluator, encryptor, encoder, relin_keys, scale, context); 
    // --- DEBUG: skip soft_step_evaluation, use constant Phi ---
    cout << " 2" << endl;
    Plaintext pt_phi_right, pt_phi_left;
    encoder.encode(0.5, scale, pt_phi_right); // or 1.0, tuỳ bạn
    encoder.encode(0.5, scale, pt_phi_left);

    cout << " 3" << endl;
    Ciphertext C_Phi_Right, C_Phi_Left;
    encryptor.encrypt(pt_phi_right, C_Phi_Right);
    encryptor.encrypt(pt_phi_left, C_Phi_Left);

    // W_new = W * Phi
    Ciphertext C_W_new_right, C_W_new_left;
    
    // Right: W * Phi_Right
    align_and_multiply(C_W_col, C_Phi_Right, C_W_new_right);
    align_and_multiply(C_W_col, C_Phi_Left, C_W_new_left);
    
    cout << "Completed compute_W_phi_best()" << endl;
    return {C_W_new_right, C_W_new_left};
}

int NUM_LABELS = 3; // Iris-setosa, Iris-versicolor, Iris-virginica
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
    size_t num_samples
){
    cout << "train_decision_tree() depth=" << depth << endl;
    auto node = make_unique<Node>();
    const size_t NUM_FEATURES = C_X_cols.size();

    // 1. ĐIỀU KIỆN DỪNG
    if (depth >= max_depth) {
        cout << " Reached max depth. Creating leaf node." << endl;
        node->is_leaf = true;
        node->leaf_value.resize(NUM_LABELS); // Khởi tạo vector kết quả Lx1

        // Lặp qua TẤT CẢ L nhãn để tính tổng trọng số cho từng nhãn
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            cout << "  Computing leaf value for label " << l << endl;
            // 1.1. TÍNH TỔNG TRỌNG SỐ ĐỒNG HÌNH (C_W * C_Y[l])
            Ciphertext C_weighted_sum = leaf_value(
                C_W_col, C_Y_cols[l], evaluator, relin_keys, galois_keys, context 
            );

            // 1.2. GIẢI MÃ (Client-side)
            Plaintext pt_result;
            decryptor.decrypt(C_weighted_sum, pt_result);
            
            vector<double> decoded_result;
            encoder.decode(pt_result, decoded_result);
            
            // 1.3. LƯU GIÁ TRỊ LÁ (Slot 0 chứa tổng cuối cùng)
            node->leaf_value[l] = decoded_result[0];
        }
        
        cout << "Created leaf node at depth=" << depth << " with values: ";
        return node; 
    }

    // 2. TÌM NGƯỠNG TỐI ƯU
    double min_gini = 1e9;
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // (Vector này sẽ lưu các giá trị W*phi(Z) cho lần gọi đệ quy sau)
    Ciphertext C_W_Phi_Best_Right, C_W_Phi_Best_Left;

    // Lặp qua TẤT CẢ thuộc tính (i) và TẤT CẢ ngưỡng (theta)
    for (int i = 0; i < 1; ++i) { // i < NUM_FEATURES
        // cout << " Feature " << i << endl;
        for (double threshold : all_thresholds) { // 163 ngưỡng // double threshold : all_thresholds
            // cout << "  Threshold " << threshold << endl;
            // 2.1. Tính tổng trọng số bảo mật (Homomorphic Counts)
            auto [C_right_counts, C_left_counts] = compute_weighted_counts_homo(
                i, threshold, C_X_cols, C_W_col, C_Y_cols, 
                evaluator, encryptor, encoder, relin_keys, galois_keys, context, scale, num_samples
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
                
                // *Lưu ý: Bạn cần tính và lưu C_W_Phi_Best_Right/Left ở đây nếu muốn tiếp tục đệ quy*

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
        relin_keys, galois_keys, context, scale, num_samples
    );

    // Nút Con Trái (Sử dụng W_new_left làm trọng số đầu vào)
    node->left_child = train_decision_tree(
        C_X_cols, C_W_new_left, C_Y_cols, all_thresholds, W_train_clear,
        depth + 1, max_depth, evaluator, encryptor, decryptor, encoder, 
        relin_keys, galois_keys, context, scale, num_samples
    );

    cout << "Completed train_decision_tree() at depth=" << depth << endl;
    return node;
}
// predic_decision_tree(x, tree)
// if tree.v is leaf_value thì return tree.v.leaf_value
// else return soft_step_evaluation(cx[v.feature-v.theta]).predic_decision_tree(x, tree.v.right) + soft_step_evaluation(cx[v.feature-v.theta]).predic_decision_tree(x, tree.v.left)

// show_tree(tree, depth)

int main()
{
    // ====== 1. Thiết lập tham số CKKS ======
    print_example_banner("1. Thiet lap tham so CKKS");
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384; // Tăng bậc đa thức để thêm chỗ cho moduli
    parms.set_poly_modulus_degree(poly_modulus_degree);

    // Chuỗi 6 moduli là đủ cho soft-step-16
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        {60, 40, 40, 40, 40, 40, 40, 40, 60}
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
    vector<Iris> data;
    try {
        data = read_iris_csv("C:/decision-tree-SEAL/Project_Build/Release/iris.csv");
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
    auto extract_data = [&](const vector<Iris>& iris_data, vector<vector<double>>& X, vector<vector<double>>& Y_onehot) {
        if (iris_data.empty()) return;
        size_t num_features = iris_data[0].features.size();
        
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

    // 5.1 Mã hóa C_W_col kich thước Nx1 (mỗi ciphertext kích thước 1x1)
    // vector<double> W_train(NUM_SAMPLES, 1.0); // Trọng số W khởi tạo là 1.0 cho tất cả mẫu kich thước Nx1
    // vector<Ciphertext> C_W_col(NUM_SAMPLES);
    // for (size_t j = 0; j < Y_train_onehot.size() ; ++j) {
    //     // cout << "Ma hoa nhan cho dac trung W thu " << j+1 << "..." << endl;
    //     Plaintext ptw;
    //     encoder.encode(W_train[j], scale, ptw); 
    //     encryptor.encrypt(ptw, C_W_col[j]);
    // }

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
    
    // 6. Tree training và các bước tiếp theo...
    print_example_banner("6. Tree training va cac buoc tiep theo");   
    set<double> unique_thresholds_set;
    for (const auto& row : X_train) {
        for (double val : row) {
            unique_thresholds_set.insert(val);
        }
    }
    vector<double> all_thresholds(unique_thresholds_set.begin(), unique_thresholds_set.end());

    all_thresholds.resize(2); // Giảm số ngưỡng để thử nghiệm nhanh

    int max_depth = 2; 
    unique_ptr<Node> root = train_decision_tree(
        C_X_cols, C_W_col, C_Y_col, all_thresholds, W_train, 0, max_depth,
        evaluator, encryptor, decryptor, encoder, relin_keys, galois_keys, context, scale, NUM_SAMPLES
    );
    
    cout << "Huan luyen cay quyet dinh hoan tat voi do sau : " << max_depth << "" << endl;
}
