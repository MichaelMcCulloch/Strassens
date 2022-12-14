use std::arch::x86_64::{
    _mm256_add_ps, _mm256_cvtss_f32, _mm256_dp_ps, _mm256_load_ps, _mm256_loadu_ps,
    _mm256_store_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

#[cfg(target_arch = "x86_64")]
pub(crate) fn dot_product(j: &mut usize, row1: &[f32], row2: &[f32], return_result: &mut f32) {
    // Process 8 floats at a time, looping until the end of the filter is reached

    while *j + 8 < row1.len() {
        unsafe {
            // Loads the row1 into a vector of 8 floats
            let row1 = _mm256_loadu_ps(row1.as_ptr());
            // Loads the row2 into a vector of 8 floats
            let row2 = _mm256_loadu_ps(row2.as_ptr());
            // Calculates the dot product of the two vectors, with 8 bits of precision
            let result = _mm256_dp_ps(row1, row2, 0xFF);
            // Converts the result to a single float value and store it
            *return_result += _mm256_cvtss_f32(result);
        }
        *j += 8;
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn subtract_matrices(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0f32; A[0].len()]; A.len()];
    if A.len() >= 8 && A[0].len() >= 8 {
        // Use AVX instructions
        let mut i = 0;
        let mut j = 0;
        while i < A.len() && j < A[0].len() {
            unsafe {
                let mut a = _mm256_loadu_ps(A[i][j..j + 8].as_ptr());
                let mut b = _mm256_loadu_ps(B[i][j..j + 8].as_ptr());
                let mut s = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(result[i][j..j + 8].as_mut_ptr(), s);
            }
            j += 8;
            if j >= A[0].len() {
                i += 1;
                j = 0;
            }
        }
    } else {
        // Use original logic
        for i in 0..A.len() {
            for j in 0..A[0].len() {
                result[i][j] += A[i][j] - B[i][j];
            }
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn add_matrices(
    A: &Vec<Vec<f32>>,
    B: &Vec<Vec<f32>>,
    C: &Vec<Vec<f32>>,
    D: &Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0f32; A[0].len()]; A.len()];
    let block_size = 8;
    if A[0].len() < block_size {
        for i in 0..A.len() {
            for j in 0..A[0].len() {
                result[i][j] += A[i][j] + B[i][j] + C[i][j] + D[i][j];
            }
        }
    } else {
        for i in 0..A.len() {
            let mut j = 0;
            while j + block_size <= A[0].len() {
                unsafe {
                    let sum = _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_load_ps(A[i][j..j + block_size].as_ptr()),
                            _mm256_load_ps(B[i][j..j + block_size].as_ptr()),
                        ),
                        _mm256_add_ps(
                            _mm256_load_ps(C[i][j..j + block_size].as_ptr()),
                            _mm256_load_ps(D[i][j..j + block_size].as_ptr()),
                        ),
                    );
                    _mm256_store_ps(result[i][j..j + block_size].as_mut_ptr(), sum);
                }
                j += block_size;
            }
            // handle remaining elements
            while j < A[0].len() {
                result[i][j] += A[i][j] + B[i][j] + C[i][j] + D[i][j];
                j += 1;
            }
        }
    }
    result
}
#[cfg(target_arch = "x86_64")]
pub(crate) fn sum_sub_matrices(
    P1: &Vec<Vec<f32>>,
    P2: &Vec<Vec<f32>>,
    P3: &Vec<Vec<f32>>,
    P4: &Vec<Vec<f32>>,
    P5: &Vec<Vec<f32>>,
    P6: &Vec<Vec<f32>>,
    P7: &Vec<Vec<f32>>,
    P8: &Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = vec![vec![0f32; P1[0].len()]; P1.len()];
    // Add the submatrices
    for i in 0..P1.len() {
        // use AVX instructions to do 8x vectorized addition
        if P1[0].len() >= 8 {
            let mut idx: usize = 0;
            while idx <= P1[0].len() - 8 {
                unsafe {
                    let v1 = _mm256_loadu_ps(P1[i][idx..idx + 8].as_ptr());
                    let v2 = _mm256_loadu_ps(P2[i][idx..idx + 8].as_ptr());
                    let v3 = _mm256_loadu_ps(P3[i][idx..idx + 8].as_ptr());
                    let v4 = _mm256_loadu_ps(P4[i][idx..idx + 8].as_ptr());
                    let v5 = _mm256_loadu_ps(P5[i][idx..idx + 8].as_ptr());
                    let v6 = _mm256_loadu_ps(P6[i][idx..idx + 8].as_ptr());
                    let v7 = _mm256_loadu_ps(P7[i][idx..idx + 8].as_ptr());
                    let v8 = _mm256_loadu_ps(P8[i][idx..idx + 8].as_ptr());

                    let sum = _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v1, v2), v3), v4),
                                v5,
                            ),
                            v6,
                        ),
                        _mm256_add_ps(v7, v8),
                    );
                    _mm256_storeu_ps(result[i][idx..idx + 8].as_mut_ptr(), sum);
                }
                idx += 8;
            }
            // handle cases that are too small for AVX
            for j in idx..P1[0].len() {
                result[i][j] = P1[i][j]
                    + P2[i][j]
                    + P3[i][j]
                    + P4[i][j]
                    + P5[i][j]
                    + P6[i][j]
                    + P7[i][j]
                    + P8[i][j];
            }
        } else {
            // handle cases that are too small for AVX
            for j in 0..P1[0].len() {
                result[i][j] = P1[i][j]
                    + P2[i][j]
                    + P3[i][j]
                    + P4[i][j]
                    + P5[i][j]
                    + P6[i][j]
                    + P7[i][j]
                    + P8[i][j];
            }
        }
    }
    result
}
