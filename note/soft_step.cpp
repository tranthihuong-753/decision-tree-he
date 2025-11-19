soft_step_evaluation() (Binary Exponentiation Tối ưu)
Tính toán đa thức phi(z) = c0 + c1*[[z]] + c2*[[z^2]] + ...
Ciphertext soft_step_evaluation(
    const Ciphertext &encrypted_z, // z = cx - theta
    Evaluator &evaluator,
    Encryptor &encryptor,
    const CKKSEncoder &encoder,
    const RelinKeys &relin_keys,
    double scale,
    const SEALContext &context,
    const vector<double>& SOFT_STEP_COEFFICIENTS // Hệ số (Bậc 8 hoặc 16)
)
{
    const size_t NUM_COEFFS = SOFT_STEP_COEFFICIENTS.size();
    const size_t DEGREE = NUM_COEFFS - 1;

    // --- BƯỚC 1: TÍNH CÁC LŨY THỪA CƠ SỞ (BASE POWERS) ---
    // base_powers[bit] = z^(2^bit)
    
    vector<Ciphertext> base_powers;
    base_powers.push_back(encrypted_z); // base_powers[0] = z^1

    Ciphertext current_z_power = encrypted_z;
    for (size_t i = 1; (1 << i) <= DEGREE; ++i) { 
        // 1. Nhân bình phương: (z^k) * (z^k)
        Ciphertext tmp;
        evaluator.multiply(current_z_power, current_z_power, tmp);
        evaluator.relinearize_inplace(tmp, relin_keys);
        evaluator.rescale_to_next_inplace(tmp); // Tiêu tốn 1 level

        base_powers.push_back(tmp);
        current_z_power = tmp; // Chuẩn bị cho lần nhân tiếp theo
    }
    
    // --- BƯỚC 2: TÍNH TỔNG ĐA THỨC (POLYNOMIAL SUM) ---
    
    // 2.1. Khởi tạo Result bằng c0 (Sử dụng phép cộng Plaintext an toàn)
    Plaintext c0_plain;
    encoder.encode(SOFT_STEP_COEFFICIENTS[0], scale, c0_plain);

    Ciphertext result;
    // Khởi tạo result là Ciphertext Zero (bằng cách negate rồi add_plain)
    result = encrypted_z;
    evaluator.negate_inplace(result);
    evaluator.add_inplace(result, encrypted_z); 
    // Giờ result là Ciphertext Zero, ta add c0 vào
    evaluator.add_plain(result, c0_plain, result);
    
    // Mod-switch result về level cuối cùng (level thấp nhất)
    if (!base_powers.empty()) {
        evaluator.mod_switch_to_inplace(result, base_powers.back().parms_id()); 
    }

    // 2.2. Cộng dồn các số hạng (sử dụng Binary Exponentiation)
    for (size_t i = 1; i <= DEGREE; ++i) {
        double coeff = SOFT_STEP_COEFFICIENTS[i];
        if (std::abs(coeff) < 1e-12) continue; // Bỏ qua hệ số zero

        // Xây dựng z^i từ tổ hợp các base_powers
        Ciphertext C_Zi;
        bool C_Zi_initialized = false;
        
        for (int bit = 0; (1 << bit) <= (int)i; ++bit) { 
            if ((i >> bit) & 1) { // Nếu bit thứ 'bit' là 1
                
                const Ciphertext& C_Base_Power = base_powers[bit]; 
                
                if (!C_Zi_initialized) {
                    C_Zi = C_Base_Power;
                    C_Zi_initialized = true;
                } else {
                    // NHÂN TỔ HỢP: C_Zi_old * C_Base_Power
                    Ciphertext tmp_new;
                    Ciphertext base_copy = C_Base_Power;
                    
                    // Căn chỉnh level trước khi nhân
                    if (base_copy.parms_id() != C_Zi.parms_id()) {
                        evaluator.mod_switch_to_inplace(base_copy, C_Zi.parms_id());
                    }

                    evaluator.multiply(C_Zi, base_copy, tmp_new);
                    evaluator.relinearize_inplace(tmp_new, relin_keys);
                    evaluator.rescale_to_next_inplace(tmp_new); 
                    
                    C_Zi = tmp_new; // Cập nhật C_Zi
                }
            }
        }
        
        // --- 2.3. Nhân Hệ số và Cộng dồn ---
        
        // Đảm bảo C_Zi có cùng level với Result (level thấp nhất)
        if (C_Zi.parms_id() != result.parms_id()) {
            evaluator.mod_switch_to_inplace(C_Zi, result.parms_id());
        }

        // Encode hệ số với scale của C_Zi (≈ Result)
        Plaintext coeff_plain;
        encoder.encode(coeff, C_Zi.scale(), coeff_plain);

        // Nhân: term = coeff * C_Zi
        Ciphertext term;
        evaluator.multiply_plain(C_Zi, coeff_plain, term);
        
        // SỬA LỖI SCALE: Bắt buộc gán scale trước khi cộng
        term.scale() = result.scale(); 

        evaluator.add_inplace(result, term);
    }

    cout << "Completed soft_step_evaluation() (Optimized)" << endl;
    return result;
}