#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace seal;

int main()
{
    // 1️⃣ Khởi tạo tham số CKKS
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = pow(2, 14);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60})); // Độ sâu nhân ~3
    double scale = pow(2.0, 40);

    SEALContext context(parms);

    // 2️⃣ Sinh khóa
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    // 3️⃣ Tạo các đối tượng tiện ích
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    // 4️⃣ Dữ liệu đầu vào (2 vector cần nhân)
    // vector<double> x = {1.1, 2.2, 3.3};
    double x = 1.1;
    vector<double> y = {10.0, 20.0, 30.0};
    double z = 2.2; 

    // 5️⃣ Mã hóa plaintext -> ciphertext
    Plaintext plain_x, plain_y, plain_z;
    encoder.encode(x, scale, plain_x);
    encoder.encode(y, scale, plain_y);
    encoder.encode(z, scale, plain_z);

    Ciphertext enc_x, enc_y;
    encryptor.encrypt(plain_x, enc_x);
    encryptor.encrypt(plain_y, enc_y);
    
    for (int step = 1; step < 3; step += 1) {
        Ciphertext C_Rotated;
        evaluator.rotate_vector(enc_y, step, galois_keys, C_Rotated);
        evaluator.add_inplace(enc_y, C_Rotated);
    }

    // // 6️⃣ Nhân hai ciphertext
    // Ciphertext enc_resultt;
    // evaluator.multiply(enc_x, enc_y, enc_resultt);       // Nhân
    // evaluator.relinearize_inplace(enc_resultt, relin_keys); // Thu gọn ciphertext
    // evaluator.rescale_to_next_inplace(enc_resultt);        // Rescale để giảm scale

    // cout << "Sau nhan: " << endl;
    // Ciphertext enc_result;
    // evaluator.mod_switch_to_inplace(plain_z, enc_resultt.parms_id());
    // evaluator.multiply_plain(enc_resultt, plain_z, enc_result);       // Nhân
    // evaluator.relinearize_inplace(enc_result, relin_keys); // Thu gọn ciphertext
    // evaluator.rescale_to_next_inplace(enc_result);        // Rescale để giảm scale

    // 7️⃣ Giải mã kết quả
    Plaintext plain_result;
    decryptor.decrypt(enc_y, plain_result);

    vector<double> result;
    encoder.decode(plain_result, result);

    // 8️⃣ In kết quả
    cout << "Ket qua: " << endl;
    for (size_t i = 0; i < 5; i++)
        // cout << x[i] << " * " << y[i] << " = " << result[i] << endl;
        cout << result[i] << endl;

    return 0;
}


