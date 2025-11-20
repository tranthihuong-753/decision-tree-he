#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <winsock2.h>
#include <ws2tcpip.h>

using namespace std;
using namespace seal;

void send_stream(int sock, stringstream& stream) {
    string data = stream.str();
    size_t data_size = data.size();
    
    // Gửi kích thước dữ liệu trước
    send(sock, &data_size, sizeof(data_size), 0);
    // Gửi dữ liệu
    send(sock, data.data(), data_size, 0);
    
    cout << "Sent " << data_size << " bytes" << endl;
}

int main() {
    // KHỞI TẠO WINSOCK [SOCK_STREAM]
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cerr << "WSAStartup failed" << endl;
        return 1;
    }
    // Kết nối socket
    int sock = socket("10.2.11.216", SOCK_STREAM, 0);
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8080);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    connect(sock, (sockaddr*)&server_addr, sizeof(server_addr));
    cout << "Connected to server" << endl;

    // Setup streams
    stringstream parms_stream;
    stringstream data_stream;
    stringstream sk_stream;

    // 1️⃣ Khởi tạo tham số CKKS
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384; // 2^14
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 60}));
    double scale = pow(2.0, 40);

    SEALContext context(parms);

    // Serialize parameters
    auto size = parms.save(parms_stream);
    cout << "EncryptionParameters: wrote " << size << " bytes" << endl;

    // 2️⃣ Sinh khóa
    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    secret_key.save(sk_stream);
    
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    // 3️⃣ Tạo các đối tượng tiện ích
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    // 4️⃣ Dữ liệu đầu vào
    vector<double> x = {1.1, 2.2, 3.3, 4.4, 5.5};
    vector<double> y = {10.0, 20.0, 30.0, 40.0, 50.0};

    // 5️⃣ Mã hóa plaintext -> ciphertext
    Plaintext plain_x, plain_y;
    encoder.encode(x, scale, plain_x);
    encoder.encode(y, scale, plain_y);

    Ciphertext enc_x, enc_y;
    encryptor.encrypt(plain_x, enc_x);
    encryptor.encrypt(plain_y, enc_y);

    // Serialize keys và ciphertexts
    relin_keys.save(data_stream);
    galois_keys.save(data_stream);
    enc_x.save(data_stream);
    enc_y.save(data_stream);

    // 6️⃣ Gửi dữ liệu qua socket
    send_stream(sock, parms_stream);   // Gửi parameters
    send_stream(sock, data_stream);    // Gửi keys và ciphertexts
    send_stream(sock, sk_stream);      // Gửi secret key

    cout << "All data sent to server" << endl;

    // 7️⃣ Nhận kết quả từ server
    size_t result_size;
    recv(sock, &result_size, sizeof(result_size), 0);
    
    vector<char> result_buffer(result_size);
    recv(sock, result_buffer.data(), result_size, 0);
    
    stringstream result_stream;
    result_stream.write(result_buffer.data(), result_size);

    // 8️⃣ Giải mã kết quả
    Ciphertext encrypted_result;
    encrypted_result.load(context, result_stream);

    Plaintext plain_result;
    decryptor.decrypt(encrypted_result, plain_result);

    vector<double> result;
    encoder.decode(plain_result, result);

    // 9️⃣ In kết quả
    cout << "Ket qua tinh toan: " << endl;
    for (size_t i = 0; i < min(result.size(), x.size()); i++) {
        cout << x[i] << " * " << y[i] << " = " << result[i] << endl;
    }

    close(sock);
    return 0;
}