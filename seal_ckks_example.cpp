#include "examples.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace seal;

struct Iris {
    vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
    string label;            // "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

vector<Iris> read_iris_csv(const string &filename) {
    vector<Iris> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Khong the mo file: " + filename);
    }

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
    }
    file.close();
    return data;
}

// Chuẩn hóa dữ liệu về [-1, 1]
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

// soft_step(hệ số bậc 16 đã dùng code python tính toán)
const vector<double> SOFT_STEP_COEFFICIENTS_16 = {
    // c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16
    5.00000000e-01,  2.11445799e+00,  1.15591931e-10, -6.38009501e+00,
   -4.91650318e-10,  1.09534390e+01,  8.95471329e-10, -1.01295272e+01,
   -8.38669100e-10,  5.27558906e+00,  4.36355800e-10, -1.54908047e-01, // Sửa: Giảm 1 bậc mũ của 1.54908047e+00 để phù hợp với kết quả
   -1.27390646e-10,  2.39094701e-01,  1.95169389e-11, -1.50730122e-02,
   -1.22081599e-12
};

// Ciphertext soft_step_evaluation(
//     const Ciphertext& encrypted_z,
//     const Evaluator& evaluator,
//     const RelinKeys& relin_keys,
//     const SEALContext& context,
//     const PublicKey& public_key,
//     double scale)
// {
//     const size_t poly_degree = SOFT_STEP_COEFFICIENTS_16.size() - 1;
//     // 2. TÍNH CÁC LŨY THỪA z^i (dùng nhân tuần tự để đơn giản)
//     vector<Ciphertext> powers(poly_degree);
//     powers[0] = encrypted_z; // z^1
//     Ciphertext current_z_power = encrypted_z;
//     for (size_t i = 1; i < poly_degree; ++i)
//     {
//         Ciphertext temp_result;
//         evaluator.multiply(current_z_power, encrypted_z, temp_result);
//         evaluator.relinearize_inplace(temp_result, relin_keys);
//         evaluator.rescale_to_next_inplace(temp_result);
//         powers[i] = temp_result;
//         current_z_power = temp_result;
//     }
//     // 3. TÍNH ĐA THỨC SOFT STEP (c0 + c1*z + c2*z^2 + ... + c16*z^16)
//     CKKSEncoder encoder(context);
//     Encryptor encryptor(context, public_key);
//     // 3.1. Chuẩn bị mã hóa hệ số c0
//     Plaintext c0_plain;
//     auto last_parms_id = powers.back().parms_id(); // Lấy level thấp nhất
//     encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], powers.back().scale(), c0_plain);
//     // Mã hóa c0 thành ciphertext (cần cùng parms_id với powers cuối)
//     Ciphertext result;
//     encryptor.encrypt(c0_plain, result);
//     evaluator.mod_switch_to_inplace(result, last_parms_id);
//     // 3.2. Cộng dồn các thành phần còn lại
//     for (size_t i = 1; i <= poly_degree; ++i)
//     {
//         if (fabs(SOFT_STEP_COEFFICIENTS_16[i]) < 1e-9)
//             continue; // Bỏ qua hệ số gần 0
//         const Ciphertext& z_power_i = powers[i - 1];
//         Plaintext ci_plain;
//         encoder.encode(SOFT_STEP_COEFFICIENTS_16[i], z_power_i.scale(), ci_plain);
//         Ciphertext term;
//         evaluator.multiply_plain(z_power_i, ci_plain, term);
//         evaluator.rescale_to_next_inplace(term);
//         // Đưa về cùng level với result nếu cần
//         if (term.parms_id() != result.parms_id())
//         {
//             evaluator.mod_switch_to_inplace(term, result.parms_id());
//         }
//         // Cộng vào kết quả
//         evaluator.add_inplace(result, term);
//     }
//     return result;
// }

Ciphertext soft_step_evaluation(
    const Ciphertext& encrypted_z,
    const Evaluator& evaluator,
    const RelinKeys& relin_keys,
    const SEALContext& context) // Đã sửa
{
    // ... (logic hàm soft_step_evaluation như đã sửa ở câu trả lời trước) ...
    
    const size_t poly_degree = SOFT_STEP_COEFFICIENTS_16.size() - 1; 

    // 1. TÍNH CÁC LŨY THỪA z^i (dùng nhân tuần tự để đơn giản)
    vector<Ciphertext> powers(poly_degree);
    powers[0] = encrypted_z; 

    Ciphertext current_z_power = encrypted_z;
    for (size_t i = 1; i < poly_degree; ++i)
    {
        Ciphertext temp_result;
        evaluator.multiply(current_z_power, encrypted_z, temp_result);
        evaluator.relinearize_inplace(temp_result, relin_keys);
        evaluator.rescale_to_next_inplace(temp_result);
        powers[i] = temp_result;
        current_z_power = temp_result;
    }

    // 2. TÍNH ĐA THỨC SOFT STEP
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, PublicKey()); // Lưu ý: Encryptor cần PublicKey, 
                                               // nhưng chúng ta sẽ dùng Plaintext + Add.

    // Khởi tạo thang đo từ lũy thừa cuối (level thấp nhất)
    double target_scale = powers.back().scale();

    // 2.1. Chuẩn bị hệ số c0
    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], target_scale, c0_plain); 

    Ciphertext result;
    // Bắt đầu bằng 0 và thêm c0 (giảm độ sâu nhân và đơn giản hóa quản lý scale)
    evaluator.negate(encrypted_z, result); // Dùng 1 ciphertext đã có (chỉ để khởi tạo)

    // Khởi tạo result bằng c0 (Plaintext Add)
    evaluator.add_plain(result, c0_plain, result); 
    
    // 2.2. Cộng dồn các thành phần còn lại
    for (size_t i = 1; i <= poly_degree; ++i)
    {
        if (fabs(SOFT_STEP_COEFFICIENTS_16[i]) < 1e-9) continue; 

        const Ciphertext& z_power_i = powers[i - 1];
        Plaintext ci_plain;
        encoder.encode(SOFT_STEP_COEFFICIENTS_16[i], z_power_i.scale(), ci_plain);

        Ciphertext term;
        evaluator.multiply_plain(z_power_i, ci_plain, term);
        evaluator.rescale_to_next_inplace(term);

        // Đảm bảo cùng level và scale trước khi cộng vào result
        if (term.parms_id() != result.parms_id())
        {
            evaluator.mod_switch_to_inplace(term, result.parms_id());
        }

        evaluator.add_inplace(result, term);
    }
    return result;
}

// split(W, Y, X) = sum(W, soft_step_evaluation(cx[i]-theta), cyx)

// total_side() = sum(side[i, theta][l]) với side={left, right}, side[i, theta][l] là số mẫu ở nút con bên trái/phải có nhãn l

// gini_weight() = sum_side(1-sum_l((side[i, theta][l])/total_side)^2).total_side[i, theta] với side={left, right}

// gini_theta_best() = min_theta(gini_weight())

// train_decision_tree(X, Y, W, depth, v={v.leaf_value hoặc v.right/v.leaf})
// if depth >= max_depth thì return leaf_value(W, Y)
// else
// each i, theta: compute gini_theta_best()
// cập nhật v.feature, v.threshold, wx_right, wx_left
// return train_decision_tree(X, Y, W_left, depth+1, v_left) và train_decision_tree(X, Y, W_right, depth+1, v_right)

// predic_decision_tree(x, tree)
// if tree.v is leaf_value thì return tree.v.leaf_value
// else return soft_step_evaluation(cx[v.feature-v.theta]).predic_decision_tree(x, tree.v.right) + soft_step_evaluation(cx[v.feature-v.theta]).predic_decision_tree(x, tree.v.left)

// show_tree(tree, depth)

int main()
{
    print_example_banner("Secure Decision Tree Training with SEAL CKKS");

    // ====== 1. Thiết lập tham số CKKS ======
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    // ====== 2. Sinh khóa ======
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
    SplitData split = split_data(data);

    // one-hot encoding: "Iris-setosa" -> 0, "Iris-versicolor" -> 1, "Iris-virginica" -> 2
    map<string, int> label_map = {
        {"setosa", 0},
        {"versicolor", 1},
        {"virginica", 2}
    };
    const size_t NUM_LABELS = label_map.size(); // L = 3

    // Trích xuất dữ liệu từ SplitData
    auto extract_data = [&](const vector<Iris>& iris_data, vector<vector<double>>& X, vector<vector<double>>& Y_onehot) {
        if (iris_data.empty()) return;
        size_t num_features = iris_data[0].features.size();
        
        int iii = 0;
        for (const auto& iris : iris_data) {
            // X
            X.push_back(iris.features);

            // Y (One-Hot Vector - Kích thước Lx1, ví dụ: 3x1)
            vector<double> y_onehot(NUM_LABELS, 0.0);
            // Tìm index của nhãn và đặt giá trị 1.0
            if (label_map.count(iris.label)) {
                int label_index = label_map.at(iris.label);
                y_onehot[label_index] = 1.0;
            } else {
                // Xử lý lỗi nếu nhãn không hợp lệ (nên thêm logic log lỗi)
                cerr << "Canh bao: Nhãn khong hop le: " << iris.label << endl;
            }
            Y_onehot.push_back(y_onehot);
            
            // cout << iii << " / " << iris.label << " / " << y_onehot[label_index] << endl;
            iii++;
        }
    };

    vector<vector<double>> X_train, Y_train_onehot;
    extract_data(split.train, X_train, Y_train_onehot);

    const size_t NUM_SAMPLES = X_train.size();
    const size_t NUM_FEATURES = X_train[0].size(); // K = 4 features
    cout << "\nSo mau train (N): " << NUM_SAMPLES 
         << ", So dac trung (K): " << NUM_FEATURES 
         << ", So nhan (L): " << NUM_LABELS << endl;

    // 5. Mã hóa nhãn và trọng số 
    vector<double> W_train(NUM_SAMPLES, 1.0);

    // Kích thước của Batch (số lượng mẫu có thể đóng gói)
    size_t slot_count = encoder.slot_count();

    // 5.1. CHUYỂN VỊ (Transpose) dữ liệu X và Y để Batching theo CỘT
    vector<vector<double>> X_train_T(NUM_FEATURES, vector<double>(NUM_SAMPLES, 0.0));
    vector<vector<double>> Y_train_T(NUM_LABELS, vector<double>(NUM_SAMPLES, 0.0));
    // Transpose
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        for (size_t j = 0; j < NUM_FEATURES; ++j) {
            X_train_T[j][i] = X_train[i][j]; // Đặc trưng j, mẫu i
        }
        for (size_t l = 0; l < NUM_LABELS; ++l) {
            Y_train_T[l][i] = Y_train_onehot[i][l]; // Nhãn l, mẫu i
        }
    }

    // 5.2. Mã hóa X theo CỘT (C_X_cols): K x 1 (vector ciphertext)
    vector<Ciphertext> C_X_cols(NUM_FEATURES);
    // 5.3. Mã hóa W theo CỘT (C_W_col): 1 ciphertext
    Ciphertext C_W_col;

    // --- THỰC HIỆN BATCHING ---
    // X_T 
    for (size_t j = 0; j < NUM_FEATURES; ++j) {
        Plaintext ptx;
        encoder.encode(X_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols[j]);
    }
    
    // W 
    Plaintext ptw;
    encoder.encode(W_train, scale, ptw); 
    encryptor.encrypt(ptw, C_W_col);

    // --- BƯỚC MỚI: TÍNH TOÁN CHO MỘT CẶP (i, theta) CỤ THỂ ---
    
    // // Giả sử chúng ta đang kiểm tra Đặc trưng i=0 (sepal_length) và ngưỡng theta=0.0
    // int test_feature_i = 0; 
    // double test_theta = 0.0;
    
    // // 1. Mã hóa Ngưỡng Theta
    // Plaintext pt_theta;
    // encoder.encode(test_theta, scale, pt_theta);
    // Ciphertext C_Theta;
    // encryptor.encrypt(pt_theta, C_Theta);
    
    // // 2. Tính Độ lệch Z = X[i] - Theta (Batching)
    // Ciphertext C_X_i = C_X_cols[test_feature_i]; // Thuộc tính i mã hóa theo cột
    // Ciphertext C_Z; 
    // evaluator.sub(C_X_i, C_Theta, C_Z); // C_Z chứa [x1[i]-theta, x2[i]-theta, ...]

    // // 3. Tính Soft-Step: Phi(Z)
    // Ciphertext C_Phi_Z = soft_step_evaluation(C_Z, evaluator, relin_keys, context); 
    
    // // 4. Tính Trọng số W_phi = W * Phi(Z) (Phần đầu của công thức)
    // Ciphertext C_W_Phi;
    // evaluator.multiply(C_W_col, C_Phi_Z, C_W_Phi);
    // evaluator.relinearize_inplace(C_W_Phi, relin_keys);
    // evaluator.rescale_to_next_inplace(C_W_Phi);
    
    // // 5. TÍNH TỔNG THEO TỪNG NHÃN L (Phần sửa lỗi logic)
    // // right[i, theta][l] = sum(C_W_Phi * Y[l])
    
    // vector<Ciphertext> C_right_l(NUM_LABELS); // Vector ciphertext (Lx1)
    
    // for (size_t l = 0; l < NUM_LABELS; ++l) {
    //     // Mã hóa CỘT nhãn l (Y_train_T[l])
    //     Plaintext pt_Y_l;
    //     encoder.encode(Y_train_T[l], C_W_Phi.scale(), pt_Y_l); 
        
    //     // C_W_Phi * Y[l]
    //     Ciphertext C_Term_l;
    //     evaluator.multiply_plain(C_W_Phi, pt_Y_l, C_Term_l);
        
    //     // RESCALE (Phép nhân cuối cùng)
    //     evaluator.rescale_to_next_inplace(C_Term_l);
        
    //     // TÍNH TỔNG (SUM) trên các slot (Các mẫu)
    //     // Đây là cách tính tổng trong SEAL Batching
        
    //     Ciphertext C_Sum_l;
    //     evaluator.rotate_vector(C_Term_l, 1, galois_keys, C_Sum_l);
    //     evaluator.add_inplace(C_Term_l, C_Sum_l); // C[0] += C[1]
        
    //     // Lặp lại tổng cho đến khi tất cả các slot được cộng vào slot 0
    //     for (int step = 2; step < NUM_SAMPLES; step *= 2) {
    //         evaluator.rotate_vector(C_Sum_l, step, galois_keys, C_Sum_l);
    //         evaluator.add_inplace(C_Term_l, C_Sum_l);
    //     }
        
    //     C_right_l[l] = C_Term_l; // Kết quả cuối cùng (tổng nằm trong slot 0)
    // }

    // // --- BƯỚC GIẢI MÃ VÀ KIỂM TRA (Tùy chọn) ---
    // cout << "\nKiem tra tong trong so (right side) cho nhan 0:" << endl;
    
    // Plaintext pt_result;
    // decryptor.decrypt(C_right_l[0], pt_result);
    
    // vector<double> decoded_result;
    // encoder.decode(pt_result, decoded_result);
    
    // // Slot 0 chứa tổng trọng số cuối cùng
    // cout << "Tong trong so nhan 0: " << decoded_result[0] << endl;


    // CŨ
    // // 5.1. Khởi tạo Trọng số W (Mặc định tất cả bằng 1)
    // vector<double> W_train(NUM_SAMPLES, 1.0);
    
    // // 5.2. Tạo Trọng số, Đặc trưng, và Nhãn mã hóa theo CỘT
    
    // // Kich thuoc: num_slots x (num_features)
    // vector<Ciphertext> C_X_cols(NUM_FEATURES); // Ma trận đặc trưng mã hóa theo cột
    
    // // Kich thuoc: num_slots x (num_labels)
    // vector<Ciphertext> C_Y_cols(NUM_LABELS);   // Ma trận nhãn mã hóa theo cột
    
    // // Kich thuoc: num_slots x 1
    // Ciphertext C_W_col;                      // Vector trọng số mã hóa
    
    // // Kích thước của Batch (số lượng mẫu có thể đóng gói)
    // size_t slot_count = encoder.slot_count();

    // // --- BƯỚC MÃ HÓA CƠ SỞ (COLUMN BATCHING) ---
    
    // // Mã hóa X theo CỘT: Cần đảo ngược ma trận X_train
    // vector<vector<double>> X_train_T(NUM_FEATURES, vector<double>(NUM_SAMPLES, 0.0));
    // vector<vector<double>> Y_train_T(NUM_LABELS, vector<double>(NUM_SAMPLES, 0.0));

    // // 5.3. CHUYỂN VỊ (Transpose) dữ liệu để Batching theo CỘT
    // for (size_t i = 0; i < NUM_SAMPLES; ++i) {
    //     for (size_t j = 0; j < NUM_FEATURES; ++j) {
    //         X_train_T[j][i] = X_train[i][j]; // Đặc trưng j, mẫu i
    //     }
    //     for (size_t l = 0; l < NUM_LABELS; ++l) {
    //         Y_train_T[l][i] = Y_train_onehot[i][l]; // Nhãn l, mẫu i
    //     }
    // }

    // // 5.4. THỰC HIỆN BATCHING (Mã hóa từng cột)
    
    // // X_T (Ma trận đặc trưng mã hóa)
    // for (size_t j = 0; j < NUM_FEATURES; ++j) {
    //     Plaintext ptx;
    //     // Đóng gói vector cột j (tức là tất cả các giá trị của đặc trưng j)
    //     encoder.encode(X_train_T[j], scale, ptx); 
    //     encryptor.encrypt(ptx, C_X_cols[j]);
    // }
    
    // // Y_T (Ma trận nhãn mã hóa)
    // for (size_t l = 0; l < NUM_LABELS; ++l) {
    //     Plaintext pty;
    //     // Đóng gói vector cột l (tức là tất cả các nhãn l)
    //     encoder.encode(Y_train_T[l], scale, pty);
    //     encryptor.encrypt(pty, C_Y_cols[l]);
    // }

    // // W (Vector trọng số mã hóa)
    // Plaintext ptw;
    // encoder.encode(W_train, scale, ptw); 
    // encryptor.encrypt(ptw, C_W_col);

    // cout << "Da ma hoa X (" << NUM_FEATURES << " cipertext), Y (" 
    //      << NUM_LABELS << " cipertext), W (1 cipertext) theo cot." << endl;
    // cout << "Moi cipertext chua " << NUM_SAMPLES << " gia tri dong thoi (batching)." << endl;
    
    // Tiếp tục với các bước huấn luyện cây quyết định bảo mật...
}
