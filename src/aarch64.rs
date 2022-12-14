use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vfmaq_f32, vgetq_lane_f32, vld1q_f32, vst1q_f32};

#[cfg(target_arch = "aarch64")]
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
        // use NEON instructions to do 8x vectorized addition
        if P1[0].len() >= 8 {
            let mut idx: usize = 0;
            while idx <= P1[0].len() - 8 {
                unsafe {
                    let v1 = vld1q_f32(P1[i][idx..idx + 8].as_ptr());
                    let v2 = vld1q_f32(P2[i][idx..idx + 8].as_ptr());
                    let v3 = vld1q_f32(P3[i][idx..idx + 8].as_ptr());
                    let v4 = vld1q_f32(P4[i][idx..idx + 8].as_ptr());
                    let v5 = vld1q_f32(P5[i][idx..idx + 8].as_ptr());
                    let v6 = vld1q_f32(P6[i][idx..idx + 8].as_ptr());
                    let v7 = vld1q_f32(P7[i][idx..idx + 8].as_ptr());
                    let v8 = vld1q_f32(P8[i][idx..idx + 8].as_ptr());

                    let sum = vaddq_f32(
                        vaddq_f32(
                            vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(v1, v2), v3), v4), v5),
                            v6,
                        ),
                        vaddq_f32(v7, v8),
                    );
                    vst1q_f32(result[i][idx..idx + 8].as_mut_ptr(), sum);
                }
                idx += 8;
            }
            // handle cases that are too small for NEON
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
            // handle cases that are too small for NEON
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

#[cfg(target_arch = "aarch64")]
pub(crate) fn add_matrices(
    A: &Vec<Vec<f32>>,
    B: &Vec<Vec<f32>>,
    C: &Vec<Vec<f32>>,
    D: &Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0f32; A[0].len()]; A.len()];
    if A.len() < 4 {
        for i in 0..A.len() {
            for j in 0..A[0].len() {
                result[i][j] += A[i][j] + B[i][j] + C[i][j] + D[i][j];
            }
        }
    } else {
        for i in 0..A.len() {
            for j in 0..A[0].len() {
                unsafe {
                    let sum = vaddq_f32(
                        vaddq_f32(
                            vld1q_f32(A[i][j..j + 4].as_ptr()),
                            vld1q_f32(B[i][j..j + 4].as_ptr()),
                        ),
                        vaddq_f32(
                            vld1q_f32(C[i][j..j + 4].as_ptr()),
                            vld1q_f32(D[i][j..j + 4].as_ptr()),
                        ),
                    );
                    vst1q_f32(result[i][j..j + 4].as_mut_ptr(), sum);
                }
            }
        }
    }
    result
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn subtract_matrices(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    use std::arch::aarch64::vsubq_f32;

    let mut result = vec![vec![0f32; A[0].len()]; A.len()];
    if A.len() > 4 && A[0].len() > 4 {
        // Use NEON instructions
        unsafe {
            let mut a = vld1q_f32(&A[0][0]);
            let mut b = vld1q_f32(&B[0][0]);
            let mut s = vsubq_f32(a, b);
            vst1q_f32(&mut result[0][0], s);
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
#[cfg(target_arch = "aarch64")]
pub(crate) fn dot_product(j: &mut usize, row1: &[f32], row2: &[f32], return_result: &mut f32) {
    // Process 4 floats at a time, looping until the end of the filter is reached

    while *j + 4 < row1.len() {
        unsafe {
            // Loads the row1 into a vector of 4 floats
            let row1 = vld1q_f32(row1.as_ptr());
            // Loads the row2 into a vector of 4 floats
            let row2 = vld1q_f32(row2.as_ptr());
            // Calculates the fused multiply-add (FMA) of the two vectors, with 4 bits of precision
            let result = vfmaq_f32(vdupq_n_f32(0.0), row1, row2);
            // Gets the first lane from the result vector and adds it to result
            *return_result += vgetq_lane_f32(result, 0);
        }
        *j += 4;
    }
}
