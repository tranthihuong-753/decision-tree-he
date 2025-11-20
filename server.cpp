#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <sys/socket.h> // linux 
#include <arpa/inet.h>
#include <unistd.h>
#include <winsock2.h>
#include <ws2tcpip.h>

using namespace std;
using namespace seal;

#pragma comment(lib, "ws2_32.lib")

stringstream receive_stream(int sock) {
    size_t data_size;
    recv(sock, &data_size, sizeof(data_size), 0);
    
    vector<char> buffer(data_size);
    recv(sock, buffer.data(), data_size, 0);
    
    stringstream stream;
    stream.write(buffer.data(), data_size);
    return stream;
}

int main()
{
    // KHỞI TẠO WINSOCK [SOCK_STREAM]
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cerr << "WSAStartup failed" << endl;
        return 1;
    }

    // Tạo socket server
    int server_fd = socket("10.2.11.216", SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);
    
    bind(server_fd, (sockaddr*)&address, sizeof(address));
    listen(server_fd, 3);
    
    cout << "Server listening on port 8080..." << endl;
    
    int client_sock = accept(server_fd, nullptr, nullptr);
    cout << "Client connected" << endl;
    
    // Nhận dữ liệu từ client
    auto parms_stream = receive_stream(client_sock);
    auto data_stream = receive_stream(client_sock);
    auto sk_stream = receive_stream(client_sock);

    // 1️⃣ Load parameters và tạo context
    EncryptionParameters parms;
    parms.load(parms_stream);
    SEALContext context(parms);

    // 2️⃣ Load keys và ciphertexts
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    Ciphertext enc_x, enc_y;

    relin_keys.load(context, data_stream);
    galois_keys.load(context, data_stream);
    enc_x.load(context, data_stream);
    enc_y.load(context, data_stream);

    cout << "Received all data from client" << endl;

    // 3️⃣ Thực hiện tính toán
    Evaluator evaluator(context);
    
        // Nhân hai ciphertext
    Ciphertext encrypted_prod;
    evaluator.multiply(enc_x, enc_y, encrypted_prod);
    evaluator.relinearize_inplace(encrypted_prod, relin_keys);
    evaluator.rescale_to_next_inplace(encrypted_prod);

    // Tính tổng các phần tử trong vector (rotation và cộng)
    Ciphertext enc_sum = enc_x; // Copy
    for (int step = 1; step < 3; step++) {
        Ciphertext rotated;
        evaluator.rotate_vector(enc_sum, step, galois_keys, rotated);
        evaluator.add_inplace(enc_sum, rotated);
    }

        // 4️⃣ Gửi kết quả về client
    stringstream result_stream;
    encrypted_prod.save(result_stream); // Gửi kết quả phép nhân
    
    send_stream(client_sock, result_stream);

    cout << "Computation completed and result sent to client" << endl;

    close(client_sock);
    close(server_fd);

    return 0;
}


