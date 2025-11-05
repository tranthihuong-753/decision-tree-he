#include "examples.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <map>

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
    const GaloisKeys &galois_keys){
        Ciphertext P;
        evaluator.multiply(C_W_col, C_Y_col, P);
        evaluator.relinearize_inplace(P, relin_keys);
        evaluator.rescale_to_next_inplace(P);
        
        int max_offset = 1024; // conservative; adjust to encoder.slot_count()/2
        Ciphertext res = P;
        for (int step = 1; step <= max_offset; step *= 2) {
            Ciphertext tmp;
            evaluator.rotate_vector(res, step, galois_keys, tmp);
            evaluator.add_inplace(res, tmp);
        }
        return res;
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
    double scale)
{
    // 1) Tạo vector lưu các lũy thừa của z: z^1, z^2, z^3, ...
    vector<Ciphertext> powers;
    powers.reserve(SOFT_STEP_COEFFICIENTS_16.size());
    powers.push_back(encrypted_z); // z^1

    // IMPORTANT: đảm bảo encrypted_z đã ở đúng level và scale ban đầu mong muốn.

    for (size_t i = 1; i < SOFT_STEP_COEFFICIENTS_16.size(); ++i) {
        // multiply last power by encrypted_z -> z^{i+1}
        Ciphertext tmp;
        evaluator.multiply(powers.back(), encrypted_z, tmp);            // tmp scale ≈ scale^2
        evaluator.relinearize_inplace(tmp, relin_keys);                // relinearize
        evaluator.rescale_to_next_inplace(tmp);                        // rescale: giảm scale từ scale^2 -> scale

        // Sau rescale, tmp.parms_id() thay đổi. 
        // Chúng ta cần đảm bảo mọi power có cùng parms_id khi dùng cùng nhau.
        powers.push_back(tmp);
    }

    // 2) Mã hoá hệ số tự do c0 và khởi tạo result
    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], scale, c0_plain);

    Ciphertext result;
    encryptor.encrypt(c0_plain, result); // dùng Encryptor, không phải Evaluator

    // Đảm bảo result và powers[*] cùng level trước khi cộng.
    // Nếu result.parms_id() khác, mod-switch result về parms_id của một term trước khi cộng.

    for (size_t i = 1; i < SOFT_STEP_COEFFICIENTS_16.size(); ++i) {
        double coeff = SOFT_STEP_COEFFICIENTS_16[i];
        if (std::abs(coeff) < 1e-12) continue;

        // powers[i-1] là z^i
        Ciphertext &power = powers[i-1];

        // 2.1 Encode hệ số với scale phù hợp (scale cần giống với power.scale())
        // LƯU Ý: sau rescale, power.scale() gần bằng 'scale' nhưng không chắc tuyệt đối, 
        // do đó encode bằng 'power.scale()' là an toàn hơn.
        double power_scale = power.scale(); 
        Plaintext coeff_plain;
        encoder.encode(coeff, power_scale, coeff_plain);

        // 2.2 Nếu coeff_plain.parms_id() khác với power.parms_id(), mod-switch coeff_plain
        // tới parms_id của power (để multiply_plain hợp lệ).
        evaluator.mod_switch_to_inplace(coeff_plain, power.parms_id());

        // 2.3 Nhân: term = coeff * z^i
        Ciphertext term;
        evaluator.multiply_plain(power, coeff_plain, term);
        // kết quả term có cùng parms_id với power; scale của term là power.scale() * power_scale (≈ scale^2)
        evaluator.rescale_to_next_inplace(term);   // rescale để đưa scale về ~scale

        // 2.4 Điều chỉnh levels/parms_id: trước khi cộng, result và term phải cùng parms_id
        // Chọn parms_id mục tiêu là term.parms_id()
        if (result.parms_id() != term.parms_id()) {
            // mod-switch result xuống target parms_id
            evaluator.mod_switch_to_inplace(result, term.parms_id());
        }

        // 2.5 Có thể cần điều chỉnh scale (các scale có thể hơi khác do rescale).
        // Nếu scale chênh lệch nhỏ, bạn có thể chấp nhận; nếu khác nhiều, cần xử lý thêm.
        // Ở đây giả sử scale đã khớp đủ tốt.

        // 2.6 Cộng vào kết quả
        evaluator.add_inplace(result, term);
    }

    return result;
}

// Ciphertext soft_step_evaluation(
//     const Ciphertext& encrypted_z,
//     const Evaluator& evaluator,
//     const RelinKeys& relin_keys,
//     const SEALContext& context) // Đã sửa
// {
//     // ... (logic hàm soft_step_evaluation như đã sửa ở câu trả lời trước) ...
//     const size_t poly_degree = SOFT_STEP_COEFFICIENTS_16.size() - 1; 
//     // 1. TÍNH CÁC LŨY THỪA z^i (dùng nhân tuần tự để đơn giản)
//     vector<Ciphertext> powers(poly_degree);
//     powers[0] = encrypted_z; 
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
//     // 2. TÍNH ĐA THỨC SOFT STEP
//     CKKSEncoder encoder(context);
//     Encryptor encryptor(context, PublicKey()); // Lưu ý: Encryptor cần PublicKey, 
//                                                // nhưng chúng ta sẽ dùng Plaintext + Add.
//     // Khởi tạo thang đo từ lũy thừa cuối (level thấp nhất)
//     double target_scale = powers.back().scale();
//     // 2.1. Chuẩn bị hệ số c0
//     Plaintext c0_plain;
//     encoder.encode(SOFT_STEP_COEFFICIENTS_16[0], target_scale, c0_plain); 
//     Ciphertext result;
//     // Bắt đầu bằng 0 và thêm c0 (giảm độ sâu nhân và đơn giản hóa quản lý scale)
//     evaluator.negate(encrypted_z, result); // Dùng 1 ciphertext đã có (chỉ để khởi tạo)
//     // Khởi tạo result bằng c0 (Plaintext Add)
//     evaluator.add_plain(result, c0_plain, result); 
//     // 2.2. Cộng dồn các thành phần còn lại
//     for (size_t i = 1; i <= poly_degree; ++i)
//     {
//         if (fabs(SOFT_STEP_COEFFICIENTS_16[i]) < 1e-9) continue; 
//         const Ciphertext& z_power_i = powers[i - 1];
//         Plaintext ci_plain;
//         encoder.encode(SOFT_STEP_COEFFICIENTS_16[i], z_power_i.scale(), ci_plain);
//         Ciphertext term;
//         evaluator.multiply_plain(z_power_i, ci_plain, term);
//         evaluator.rescale_to_next_inplace(term);
//         // Đảm bảo cùng level và scale trước khi cộng vào result
//         if (term.parms_id() != result.parms_id())
//         {
//             evaluator.mod_switch_to_inplace(term, result.parms_id());
//         }
//         evaluator.add_inplace(result, term);
//     }
//     return result;
// }

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
    // ====== 1. Thiết lập tham số CKKS ======
    print_example_banner("1. Thiet lap tham so CKKS");
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
    vector<double> W_train(NUM_SAMPLES, 1.0); // Trọng số W khởi tạo là 1.0 cho tất cả mẫu kich thước Nx1
    vector<Ciphertext> C_W_col(NUM_SAMPLES);
    for (size_t j = 0; j < Y_train_onehot.size() ; ++j) {
        // cout << "Ma hoa nhan cho dac trung W thu " << j+1 << "..." << endl;
        Plaintext ptw;
        encoder.encode(W_train[j], scale, ptw); 
        encryptor.encrypt(ptw, C_W_col[j]);
    }

    // 5.2. Mã hóa C_X_col kích thước Kx1 (mỗi ciphertext kích thước Nx1) 
    vector<vector<double>> X_train_T(NUM_FEATURES, vector<double>(NUM_SAMPLES, 0.0)); // ma trận KxN
    // vector<vector<double>> Y_train_T(NUM_LABELS, vector<double>(NUM_SAMPLES, 0.0)); // ma trận LxN
    // Transpose
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        for (size_t j = 0; j < NUM_FEATURES; ++j) {
            X_train_T[j][i] = X_train[i][j]; // Đặc trưng j, mẫu i
        }
        // for (size_t l = 0; l < NUM_LABELS; ++l) {
        //     Y_train_T[l][i] = Y_train_onehot[i][l]; // Nhãn l, mẫu i
        // }
    }

    vector<Ciphertext> C_X_cols(NUM_FEATURES); // Kích thước K (số đặc trưng)
    for (size_t j = 0; j < NUM_FEATURES; ++j) {
        // cout << "Ma hoa feature cho dac trung X thu " << j << "..." << endl;
        Plaintext ptx;
        encoder.encode(X_train_T[j], scale, ptx); 
        encryptor.encrypt(ptx, C_X_cols[j]);
    }
    
    // Mã hóa C_Y_col kích thước Nx1 (mỗi ciphertext ở bản rõ kích thước Lx1)
    vector<Ciphertext> C_Y_col(Y_train_onehot.size()); // Kích thước ĐÚNG là N (số mẫu = 70) 
    for (size_t j = 0; j < Y_train_onehot.size() ; ++j) {
        // cout << "Ma hoa nhan cho dac trung Y thu " << j+1 << "..." << endl;
        Plaintext pty;
        encoder.encode(Y_train_onehot[j], scale, pty); 
        encryptor.encrypt(pty, C_Y_col[j]);
    }

    // // Mỗi ciphertext C_Y_cols[l] chứa cột l của ma trận nhãn (giá trị nhãn l cho tất cả N mẫu)
    // vector<Ciphertext> C_Y_cols(NUM_LABELS); // Kích thước ĐÚNG là L (số nhãn)
    // // --- THỰC HIỆN BATCHING cho Y ---
    // for (size_t l = 0; l < NUM_LABELS; ++l) {
    //     Plaintext pty;
    //     // Y_train_T[l] là cột l của ma trận nhãn (N giá trị)
    //     encoder.encode(Y_train_T[l], scale, pty); 
    //     encryptor.encrypt(pty, C_Y_cols[l]);
    // }

    // Tiếp tục với các bước huấn luyện cây quyết định bảo mật...
}
