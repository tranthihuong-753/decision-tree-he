// SAVE & LOAD KEY 
// [+] All HE parameters loaded successfully. 
// Luu y create_galois_keys
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

string src_key = "C:/hu/decision-tree-he/config/pyfhel/"; /// 
size_t poly_modulus_degree = pow(2, 15); 
vector<int> coeff_modulus_bits = 
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}; // 11x40  
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


Ciphertext rotate_number(
    const Ciphertext &ct,
    int number,
    const GaloisKeys &gal_keys,
    const Evaluator &evaluator)
{
    Ciphertext res = ct;
    Ciphertext tmp;
    int step = 1;
    int k = number;

    bool first = true;

    while (k > 0) {
        if (k & 1) {
            if (first) {
                evaluator.rotate_vector(ct, step, gal_keys, res);
                first = false;
            } else {
                evaluator.rotate_vector(res, step, gal_keys, tmp);
                res = tmp;
            }
        }
        k >>= 1;
        step <<= 1;
    }
    return res;
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
    save_parms(folder + "parms.bin", parms);
    save_key(folder + "public.key", public_key);
    save_key(folder + "relin.key", relin_keys);
    save_key(folder + "galois.key", galois_keys);
    save_key(folder + "secret.key", sec_keys);

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
// load 4 key va context luu tu seal 
tuple<shared_ptr<SEALContext>, PublicKey, RelinKeys, GaloisKeys, SecretKey>
load_all(const string &folder)
{
    EncryptionParameters parms = load_parms(folder + "parms.bin");

    // Tạo shared_ptr<SEALContext>
    auto context = std::make_shared<SEALContext>(parms);

    // Load các key
    PublicKey public_key = load_key<PublicKey>(folder + "public.key", context);
    RelinKeys relin_keys = load_key<RelinKeys>(folder + "relin.key", context);
    GaloisKeys galois_keys = load_key<GaloisKeys>(folder + "galois.key", context);
    SecretKey sec_keys = load_key<SecretKey>(folder + "secret.key", context);

    cout << "[+] All HE parameters loaded successfully." << endl;

    // return tuple đầy đủ
    return make_tuple(context, public_key, relin_keys, galois_keys, sec_keys);
}

void load_all_keys(
    const shared_ptr<SEALContext> &context,
    const std::string &pub_path,
    const std::string &sec_path,
    const std::string &relin_path,
    const std::string &galois_path,
    PublicKey &public_key,
    SecretKey &secret_key,
    RelinKeys &relin_keys,
    GaloisKeys &galois_keys
)
{
    // ---- LOAD PUBLIC ----
    {
        cout << "LOAD PUBLIC" << endl;
        std::ifstream f(pub_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open public.key");
        public_key.load(*context, f);
        cout << "DONE" << endl;
    }

    // ---- LOAD SECRET ----
    {
        cout << "LOAD SECRET" << endl;
        std::ifstream f(sec_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open secret.key");
        secret_key.load(*context, f);
        cout << "DONE" << endl;
    }

    // ---- LOAD RELIN KEYS ----
    {
        cout << "LOAD RELIN" << endl;
        std::ifstream f(relin_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open relin.key");
        try{
            relin_keys.load(*context, f);
        }catch (const std::exception &e) {
            cerr << "Error loading relin key: " << e.what() << endl;
        }
        cout << "DONE" << endl;
    }

    // ---- LOAD GALOIS KEYS ----
    {
        cout << "LOAD GALOIS" << endl;
        std::ifstream f(galois_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open galois.key");
        
        try{
            galois_keys.load(*context, f);
        }catch (const std::exception &e) {
            cerr << "Error loading relin key: " << e.what() << endl;
        }
        cout << "DONE" << endl;
    }
}

// Load 4 key va tao context khi save boi python 
tuple<shared_ptr<SEALContext>, PublicKey, RelinKeys, GaloisKeys, SecretKey>
load_all_new(const string &folder)
{
    EncryptionParameters parms(scheme_type::ckks);
    // bộ 3 tham số poly_modulus_degree và coeff_modulus_bits và scale
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree,
        coeff_modulus_bits
    ));
    // SEALContext context(parms);
    auto context = std::make_shared<SEALContext>(parms);

    // Load các key
    // PublicKey public_key = load_key<PublicKey>(folder + "public.key", context);
    // RelinKeys relin_keys = load_key<RelinKeys>(folder + "relin.key", context);
    // GaloisKeys galois_keys = load_key<GaloisKeys>(folder + "galois.key", context);
    // SecretKey sec_keys = load_key<SecretKey>(folder + "secret.key", context);
    
    PublicKey public_key;
    SecretKey sec_keys;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;

    load_all_keys(
        context,
        folder + "public.key",
        folder + "secret.key",
        folder + "relin.key",
        folder + "galois.key",
        public_key,
        sec_keys,
        relin_keys,
        galois_keys
    );

    cout << "[+] All HE parameters loaded successfully." << endl;

    // return tuple đầy đủ
    return make_tuple(context, public_key, relin_keys, galois_keys, sec_keys);
}

Ciphertext sum_slots(
    Ciphertext ct,
    size_t n,
    Evaluator &evaluator,
    const GaloisKeys &gal_keys
) {
    for (size_t step = 1; step < n; step <<= 1) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct, step, gal_keys, rotated);
        evaluator.add_inplace(ct, rotated);
    }
    return ct; // tổng nằm ở slot 0
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
    // vector<int> steps = {1};
    // keygen.create_galois_keys(steps, gal_keys);
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    auto sec_keys = keygen.secret_key();  
    cout << "1" << endl;
    Decryptor decryptor(context, sec_keys);
    cout << "2" << endl;
    CKKSEncoder encoder(context);

    // save_all(src_key, parms, public_key, relin_keys, gal_keys, sec_keys);///
    // auto[context_s, public_key_s, relin_keys_s, galois_keys_s, sec_keys_s] = load_all_new(src_key);
    // CKKSEncoder encoder_s(*context_s);
    // Decryptor decryptor_s(*context_s, sec_keys_s);
    // Evaluator evaluator_s(*context_s);
    // Encryptor encryptor_s(*context_s, public_key_s);
    
    //vecto c_v 
    // std::vector<double> v = {0.1, 0.2, 0.3, 0.4, 0.5};
    size_t slot_count = encoder.slot_count();
    vector<double> v(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        v[i] = (double)i; 
    }
    Plaintext c0_plain;
    encoder.encode(v, scale, c0_plain);
    Ciphertext c_v;
    encryptor.encrypt(c0_plain, c_v); 

    Ciphertext c_v_rot_1;
    evaluator.rotate_vector(c_v, 1, gal_keys, c_v_rot_1);
    show_cipher(c_v_rot_1, 7, decryptor, encoder);
    // // so n 
    // int x = 2; 
    // Plaintext c_plain;
    // encoder.encode(x, scale, c_plain);
    // Ciphertext c_n;
    // encryptor.encrypt(c_plain, c_n); 
    // show_cipher(c_n, 5, decryptor, encoder);

    // // c_v*c_n
    // Ciphertext out = multiply_ciphertexts(c_v, c_n, evaluator_s, relin_keys_s, *context_s);
    // // giai ma 
    // show_cipher(out, 5, decryptor_s, encoder_s);

    return 0;
}
