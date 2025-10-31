#include <iostream>
#include <vector>
#include <seal/seal.h>
#include <cmath>
#include "examples.h"
#include <numeric>
#include <filesystem>
using namespace std;
using namespace seal;

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

// Hàm tính tổng w*y mã hóa
Ciphertext compute_encrypted_leaf_value(
    const vector<Ciphertext> &Enc_weights,
    const vector<Ciphertext> &Enc_labels,
    Evaluator &evaluator,
    RelinKeys &relin_keys,
    Encryptor &encryptor,
    CKKSEncoder &encoder,
    double scale)
{
    Plaintext zero_plain;
    encoder.encode(vector<double>{0.0}, scale, zero_plain);
    Ciphertext Enc_Sum;
    encryptor.encrypt(zero_plain, Enc_Sum);

    for (size_t i = 0; i < Enc_weights.size(); i++) {
        Ciphertext prod;
        evaluator.multiply(Enc_weights[i], Enc_labels[i], prod);
        evaluator.relinearize_inplace(prod, relin_keys);
        evaluator.rescale_to_next_inplace(prod);

        evaluator.mod_switch_to_inplace(Enc_Sum, prod.parms_id());
        evaluator.add_inplace(Enc_Sum, prod);
    }
    return Enc_Sum;
}

Ciphertext polynomial_soft_step_stable(
    const Ciphertext &Enc_diff,
    const vector<double> &coeffs,
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    Encryptor &encryptor,
    RelinKeys &relin_keys,
    double scale
) {
    Ciphertext x_sq, x_cubed, result;
    Plaintext plain_tmp;

    // x^2
    evaluator.square(Enc_diff, x_sq);
    evaluator.relinearize_inplace(x_sq, relin_keys);
    evaluator.rescale_to_next_inplace(x_sq);

    // x^3 = x^2 * x
    Ciphertext x_rescaled = Enc_diff;
    evaluator.rescale_to_next_inplace(x_rescaled);
    evaluator.multiply(x_sq, x_rescaled, x_cubed);
    evaluator.relinearize_inplace(x_cubed, relin_keys);
    evaluator.rescale_to_next_inplace(x_cubed);

    // result = beta0 + beta1*x + beta2*x^2 + beta3*x^3
    encoder.encode(vector<double>{coeffs[0]}, scale, plain_tmp);
    encryptor.encrypt(plain_tmp, result);

    Plaintext p1, p2, p3;
    encoder.encode(vector<double>{coeffs[1]}, scale, p1);
    encoder.encode(vector<double>{coeffs[2]}, scale, p2);
    encoder.encode(vector<double>{coeffs[3]}, scale, p3);

    Ciphertext tmp;

    // beta1 * x
    evaluator.multiply_plain(Enc_diff, p1, tmp);
    evaluator.rescale_to_next_inplace(tmp);
    evaluator.add_inplace(result, tmp);

    // beta2 * x^2
    evaluator.multiply_plain(x_sq, p2, tmp);
    evaluator.rescale_to_next_inplace(tmp);
    evaluator.add_inplace(result, tmp);

    // beta3 * x^3
    evaluator.multiply_plain(x_cubed, p3, tmp);
    evaluator.rescale_to_next_inplace(tmp);
    evaluator.add_inplace(result, tmp);

    return result;
}


pair<Ciphertext, Ciphertext> compute_conditional_counts(
    CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator,
    RelinKeys &relin_keys, double scale,
    const vector<Ciphertext> &Enc_features_i,
    const vector<Ciphertext> &Enc_weights,
    const vector<Ciphertext> &Enc_labels,
    double theta,
    const vector<double> &soft_step_coeffs)
{
    Plaintext theta_plain;
    encoder.encode(vector<double>{theta}, scale, theta_plain);
    Ciphertext Enc_theta;
    encryptor.encrypt(theta_plain, Enc_theta);

    Ciphertext Enc_Sum_R, Enc_Sum_L;
    Plaintext zero_plain;
    encoder.encode(vector<double>{0.0}, scale, zero_plain);
    encryptor.encrypt(zero_plain, Enc_Sum_R);
    encryptor.encrypt(zero_plain, Enc_Sum_L);

    for (size_t i = 0; i < Enc_features_i.size(); i++) {
        Ciphertext Enc_diff_R, Enc_diff_L;
        evaluator.sub(Enc_features_i[i], Enc_theta, Enc_diff_R);
        evaluator.sub(Enc_theta, Enc_features_i[i], Enc_diff_L);

        Ciphertext Enc_Mask_R = polynomial_soft_step_stable(
            Enc_diff_R, soft_step_coeffs, evaluator, encoder, encryptor, relin_keys, scale);
        Ciphertext Enc_Mask_L = polynomial_soft_step_stable(
            Enc_diff_L, soft_step_coeffs, evaluator, encoder, encryptor, relin_keys, scale);

        // w * Mask
        Ciphertext Enc_wR, Enc_wL;
        evaluator.multiply(Enc_weights[i], Enc_Mask_R, Enc_wR);
        evaluator.relinearize_inplace(Enc_wR, relin_keys);
        evaluator.rescale_to_next_inplace(Enc_wR);

        evaluator.multiply(Enc_weights[i], Enc_Mask_L, Enc_wL);
        evaluator.relinearize_inplace(Enc_wL, relin_keys);
        evaluator.rescale_to_next_inplace(Enc_wL);

        // (w*Mask)*y
        Ciphertext Enc_y_lvl = Enc_labels[i];
        evaluator.rescale_to_next_inplace(Enc_y_lvl);

        Ciphertext prod_R, prod_L;
        evaluator.multiply(Enc_wR, Enc_y_lvl, prod_R);
        evaluator.multiply(Enc_wL, Enc_y_lvl, prod_L);

        evaluator.relinearize_inplace(prod_R, relin_keys);
        evaluator.relinearize_inplace(prod_L, relin_keys);
        evaluator.rescale_to_next_inplace(prod_R);
        evaluator.rescale_to_next_inplace(prod_L);

        evaluator.mod_switch_to_inplace(Enc_Sum_R, prod_R.parms_id());
        evaluator.add_inplace(Enc_Sum_R, prod_R);
        evaluator.mod_switch_to_inplace(Enc_Sum_L, prod_L.parms_id());
        evaluator.add_inplace(Enc_Sum_L, prod_L);
    }

    return {Enc_Sum_R, Enc_Sum_L};
}

Ciphertext compute_encrypted_leaf_value_fixed_SEAL(
    Encryptor &encryptor,
    Evaluator &evaluator,
    RelinKeys &relin_keys,
    vector<Ciphertext> &enc_weights,
    vector<Ciphertext> &enc_labels)
{
    if (enc_weights.empty() || enc_weights.size() != enc_labels.size())
        throw invalid_argument("Danh sach khong hop le.");

    Ciphertext enc_sum;
    evaluator.multiply_plain(enc_labels[0], Plaintext("0"), enc_sum); // khởi tạo 0
    for (size_t i = 0; i < enc_weights.size(); i++)
    {
        Ciphertext c_product;
        evaluator.multiply(enc_weights[i], enc_labels[i], c_product);
        evaluator.relinearize_inplace(c_product, relin_keys);
        evaluator.rescale_to_next_inplace(c_product);
        evaluator.add_inplace(enc_sum, c_product);
    }

    return enc_sum;
}

struct TreeNode
{
    int depth;
    bool is_leaf;
    double threshold;
    int feature_index;
    int predicted_label;
    TreeNode *left;
    TreeNode *right;
};

TreeNode *tree_train_secure(
    SEALContext &context,
    Encryptor &encryptor,
    Evaluator &evaluator,
    Decryptor &decryptor,
    RelinKeys &relin_keys,
    vector<vector<Ciphertext>> &Enc_X_mat,
    vector<Ciphertext> &Enc_Y_list,
    vector<Ciphertext> &Enc_W_list,
    vector<vector<double>> &X_plain,
    int depth,
    int max_depth)
{
    if (depth >= max_depth || Enc_W_list.size() < 2)
    {
        // Tính leaf value
        Ciphertext leaf_enc = compute_encrypted_leaf_value_fixed_SEAL(
            encryptor, evaluator, relin_keys, Enc_W_list, Enc_Y_list);

        Plaintext leaf_dec;
        decryptor.decrypt(leaf_enc, leaf_dec);

        double leaf_value = stod(leaf_dec.to_string());
        cout << string(depth * 2, ' ')
             << "Leaf depth " << depth
             << " value ≈ " << leaf_value << endl;

        TreeNode *leaf = new TreeNode();
        leaf->is_leaf = true;
        leaf->depth = depth;
        leaf->predicted_label = static_cast<int>(round(leaf_value));
        return leaf;
    }

    // (Demo) chọn feature 0, threshold = 0
    int best_feature = 0;
    double best_threshold = 0.0;

    cout << string(depth * 2, ' ')
         << "Split depth " << depth
         << " feature " << best_feature
         << " threshold " << best_threshold << endl;

    // Tách mẫu tạm thời — không thật vì chưa có soft-step
    vector<Ciphertext> W_left = Enc_W_list;
    vector<Ciphertext> W_right = Enc_W_list;

    // Đệ quy 2 nhánh
    TreeNode *node = new TreeNode();
    node->depth = depth;
    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = tree_train_secure(context, encryptor, evaluator, decryptor,
                                   relin_keys, Enc_X_mat, Enc_Y_list, W_left,
                                   X_plain, depth + 1, max_depth);
    node->right = tree_train_secure(context, encryptor, evaluator, decryptor,
                                    relin_keys, Enc_X_mat, Enc_Y_list, W_right,
                                    X_plain, depth + 1, max_depth);
    return node;
}

struct Iris {
    vector<double> features; // [sepal_length, sepal_width, petal_length, petal_width]
    string label;            // "Iris-setosa", "Iris-versicolor", "Iris-virginica"
};

vector<Iris> read_iris_csv(const string &filename)
{
    vector<Iris> data;
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Khong mo duoc file: " + filename);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue; // bỏ dòng trống

        stringstream ss(line);
        string token;
        vector<string> tokens;

        while (getline(ss, token, ',')) {
            // Xóa khoảng trắng đầu/cuối mỗi giá trị
            token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
            tokens.push_back(token);
        }

        if (tokens.size() < 5) {
            cerr << "⚠️  Dòng không đủ 5 cột, bỏ qua: " << line << endl;
            continue;
        }

        Iris iris;
        iris.features.resize(4);

        try {
            for (int i = 0; i < 4; i++) {
                iris.features[i] = stod(tokens[i]);
            }
            iris.label = tokens[4];
            data.push_back(iris);
        }
        catch (const invalid_argument &) {
            cerr << "❌ Lỗi stod (chuỗi không hợp lệ) ở dòng: " << line << endl;
        }
        catch (const out_of_range &) {
            cerr << "❌ Lỗi stod (vượt giới hạn) ở dòng: " << line << endl;
        }
    }

    file.close();
    return data;
}

vector<vector<double>> extract_features(const vector<Iris> &data)
{
    vector<vector<double>> X;
    X.reserve(data.size());
    for (auto &row : data)
        X.push_back(row.features);
    return X;
}

vector<double> extract_labels_numeric(const vector<Iris> &data)
{
    vector<double> y;
    y.reserve(data.size());
    for (auto &row : data)
    {
        if (row.label == "Iris-setosa") y.push_back(0.0);
        else if (row.label == "Iris-versicolor") y.push_back(1.0);
        else if (row.label == "Iris-virginica") y.push_back(2.0);
        else y.push_back(-1.0); // nhãn lạ
    }
    return y;
}


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

    // ====== 4. Chia dữ liệu ======
    SplitData split = split_data(data);
    vector<vector<double>> X_train, X_val, X_test;
    vector<double> Y_train, Y_val, Y_test;

    auto labels_to_numeric = [](const string &label) -> double {
        if (label == "Iris-setosa") return 0.0;
        if (label == "Iris-versicolor") return 1.0;
        if (label == "Iris-virginica") return 2.0;
        return -1.0;
    };

    for (auto &row : split.train) { X_train.push_back(row.features); Y_train.push_back(labels_to_numeric(row.label)); }
    for (auto &row : split.val)   { X_val.push_back(row.features);   Y_val.push_back(labels_to_numeric(row.label)); }
    for (auto &row : split.test)  { X_test.push_back(row.features);  Y_test.push_back(labels_to_numeric(row.label)); }

    cout << "\nSnSo mau train: " << X_train.size() 
         << ", val: " << X_val.size()
         << ", test: " << X_test.size() << endl;

    // ====== 5. Hàm mã hóa/giải mã ======
    auto encrypt_vector = [&](const vector<double> &v) {
        Plaintext p;
        encoder.encode(v, scale, p);
        Ciphertext c;
        encryptor.encrypt(p, c);
        return c;
    };

    auto decrypt_vector = [&](const Ciphertext &c) {
        Plaintext p;
        decryptor.decrypt(c, p);
        vector<double> res;
        encoder.decode(p, res);
        return res;
    };

    // ====== 6. Mã hóa nhãn và trọng số ======
    vector<Ciphertext> Enc_Y, Enc_W;
    for (double y : Y_train) {
        Enc_Y.push_back(encrypt_vector({y}));
        Enc_W.push_back(encrypt_vector({1.0})); // weight = 1
    }

    // ====== 7. Mã hóa ma trận X ======
    vector<vector<Ciphertext>> Enc_X_mat(X_train[0].size());
    for (size_t j = 0; j < X_train[0].size(); ++j)
        for (size_t i = 0; i < X_train.size(); ++i)
            Enc_X_mat[j].push_back(encrypt_vector({X_train[i][j]}));

    cout << "\n Ma hoa du lieu hoan tat." << endl;

    // ====== 8. Huấn luyện cây 2-depth demo ======
    cout << "\n Bat dau huan luyen cay ma hoa..." << endl;
    TreeNode *root = tree_train_secure(
        context, encryptor, evaluator, decryptor,
        relin_keys, Enc_X_mat, Enc_Y, Enc_W,
        X_train, 0, 4);
    cout << "\n Huan luyen hoan tat!" << endl;

    // ====== 9. Kiểm tra leaf value ======
    Ciphertext leaf_ct = compute_encrypted_leaf_value(
        Enc_W, Enc_Y, evaluator, relin_keys, encryptor, encoder, scale);
    vector<double> result = decrypt_vector(leaf_ct);

    cout << "\n Decrypted leaf vector (sum w*y): ";
    for (double v : result) cout << v << " ";
    cout << endl;

    cout << "\n KET THUC.\n";
    return 0;
}
