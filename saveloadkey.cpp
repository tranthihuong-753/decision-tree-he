// SAVE & LOAD KEY 
// [+] All HE parameters loaded successfully (PublicKey, RelinKeys, GaloisKeys, SecretKey)
// Luu y create_galois_keys
// Càn RAM >64GB +Time => Test tren server roi xem ntn 

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
#include <seal/seal.h>
#include <windows.h>
using namespace std;
using namespace seal;

string src_key = "C:/hu/decision-tree-he/config/server/"; /// 
size_t poly_modulus_degree = pow(2, 15); 
vector<int> coeff_modulus_bits = 
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40} ; // 18x40  
double scale = pow(2.0, 40);

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

void ensure_folder_exists(const string &folder)
{
    // Nếu folder chưa tồn tại → tạo mới
    CreateDirectoryA(folder.c_str(), NULL);
}

void save_parms(const string &path, const EncryptionParameters &parms)
{
    string folder = path.substr(0, path.find_last_of("/\\") + 1);
    ensure_folder_exists(folder);

    cout << "save_parms()" << endl;

    ofstream ofs(path, ios::binary);
    if (!ofs.is_open())
    {
        cout << "Cannot open file: " << path << endl;
        return;
    }

    parms.save(ofs);
    ofs.close();
    cout << "[+] Saved: " << path << endl;
}

// Lưu PublicKey / RelinKeys / GaloisKeys
template<typename T>
void save_key(const string &path, const T &key)
{
    cout << "save_key() " << path << endl;
    ofstream ofs(path, ios::binary);
    key.save(ofs);
    ofs.close();
    cout << "[+] Saved: " << path << endl;
}

// ====== SAVE TẤT CẢ ========
void save_all(
    const string &folder,
    const EncryptionParameters &parms,
    const PublicKey &public_key,
    const RelinKeys &relin_keys,
    const GaloisKeys &galois_keys,
    const SecretKey &sec_keys
){
    cout << "save_all()" << endl;
    try{
        save_parms(folder + "parms.bin", parms);
        save_key(folder + "public.key", public_key);
        save_key(folder + "relin.key", relin_keys);
        save_key(folder + "galois.key", galois_keys);
        save_key(folder + "secret.key", sec_keys);
    } catch (const std::exception &e) {
        cout << "Error during save_all(): " << e.what() << endl;
        return;
    }

    cout << "[+] All HE parameters saved successfully." << endl;
}

// Load EncryptionParameters
EncryptionParameters load_parms(const string &path)
{
    EncryptionParameters parms(scheme_type::ckks);
    ifstream ifs(path, ios::binary);
    parms.load(ifs);
    ifs.close();
    cout << "[+] Loaded: " << path << endl;
    return parms;
}

template<typename T>
T load_key(const string &path, const shared_ptr<SEALContext> &context)
{
    T key;
    ifstream ifs(path, ios::binary);
    key.load(*context, ifs);   // <-- dùng *context
    ifs.close();
    cout << "[+] Loaded: " << path << endl;
    return key;
}
tuple<shared_ptr<SEALContext>, PublicKey, RelinKeys, GaloisKeys, SecretKey>
load_all(const string &folder)
{
    EncryptionParameters parms = load_parms(folder + "parms.bin");

    // Tạo shared_ptr<SEALContext>
    auto context = std::make_shared<SEALContext>(parms);

    // Load các key
    PublicKey public_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    SecretKey sec_keys;

    try{
        public_key = load_key<PublicKey>(folder + "public.key", context);
        relin_keys = load_key<RelinKeys>(folder + "relin.key", context);
        galois_keys = load_key<GaloisKeys>(folder + "galois.key", context);
        sec_keys = load_key<SecretKey>(folder + "secret.key", context);
    }catch (const std::exception &e) {
        cout << "Error during load_all(): " << e.what() << endl;
        // Trả về tuple rỗng trong trường hợp lỗi
        return make_tuple(nullptr, PublicKey(), RelinKeys(), GaloisKeys(), SecretKey());
    }

    cout << "[+] All HE parameters loaded successfully." << endl;

    // return tuple đầy đủ
    return make_tuple(context, public_key, relin_keys, galois_keys, sec_keys);
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
    // GaloisKeys gal_keys;
    // keygen.create_galois_keys(gal_keys);
    vector<int> steps;
    for (int k = 1; k < poly_modulus_degree / 2; k++) {
        steps.push_back(k);
    }

    GaloisKeys gal_keys;
    try{
        keygen.create_galois_keys(steps, gal_keys);
    } catch (const std::exception &e) {
        cout << "Error during create_galois_keys(): " << e.what() << endl;
        return -1;
    }


    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    auto sec_keys = keygen.secret_key();  
    Decryptor decryptor(context, sec_keys);
    CKKSEncoder encoder(context);
    cout << "[+] All HE parameters created successfully." << endl;

    auto start = std::chrono::high_resolution_clock::now();
    save_all(src_key, parms, public_key, relin_keys, gal_keys, sec_keys);///
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time save_all(): " << duration << " ms" << endl;

    auto start1 = std::chrono::high_resolution_clock::now();
    auto[context_s, public_key_s, relin_keys_s, galois_keys_s, sec_keys_s] = load_all(src_key);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    cout << "Time load_all(): " << duration1 << " ms" << endl;

    CKKSEncoder encoder_s(*context_s);
    Decryptor decryptor_s(*context_s, sec_keys_s);
    Evaluator evaluator_s(*context_s);
    Encryptor encryptor_s(*context_s, public_key_s);
    
    //vecto c_v 
    std::vector<double> v = {1, 2, 3};
    Plaintext c0_plain;
    encoder_s.encode(v, scale, c0_plain);
    Ciphertext c_v;
    encryptor_s.encrypt(c0_plain, c_v); 
    // so n 
    int x = 2; 
    Plaintext c_plain;
    encoder_s.encode(x, scale, c_plain);
    Ciphertext c_n;
    encryptor_s.encrypt(c_plain, c_n); 

    // c_v*c_n
    Ciphertext out = multiply_ciphertexts(c_v, c_n, evaluator_s, relin_keys_s, *context_s);
    // giai ma 
    show_cipher(out, 5, decryptor_s, encoder_s);
    return 0;
}
