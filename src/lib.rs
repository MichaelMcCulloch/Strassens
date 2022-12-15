fn partition(
    A: &[Vec<f32>],
    B: &[Vec<f32>],
) -> (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
) {
    let n = A.len();
    let midpoint = n / 2;

    let mut A11 = vec![vec![0.0; midpoint]; midpoint];
    let mut A12 = vec![vec![0.0; midpoint]; midpoint];
    let mut A21 = vec![vec![0.0; midpoint]; midpoint];
    let mut A22 = vec![vec![0.0; midpoint]; midpoint];

    let mut B11 = vec![vec![0.0; midpoint]; midpoint];
    let mut B12 = vec![vec![0.0; midpoint]; midpoint];
    let mut B21 = vec![vec![0.0; midpoint]; midpoint];
    let mut B22 = vec![vec![0.0; midpoint]; midpoint];

    for i in 0..midpoint {
        for j in 0..midpoint {
            A11[i][j] = A[i][j];
            B11[i][j] = B[i][j];
            A12[i][j] = A[i][j + midpoint];
            B12[i][j] = B[i][j + midpoint];
            A21[i][j] = A[i + midpoint][j];
            B21[i][j] = B[i + midpoint][j];
            A22[i][j] = A[i + midpoint][j + midpoint];
            B22[i][j] = B[i + midpoint][j + midpoint];
        }
    }

    (A11, A12, A21, A22, B11, B12, B21, B22)
}
#[cfg(target_arch = "aarch64")]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut sum: f32 = 0.0;
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        unsafe {
            let a_slice = vld1q_f32(a[i..i + 4].as_ptr());
            let b_slice = vld1q_f32(b[i..i + 4].as_ptr());
            let dot_prod = vfmaq_f32(vdupq_n_f32(0.0), a_slice, b_slice);
            sum += vgetq_lane_f32(dot_prod, 0);
        }
        i += 4;
    }

    // Process remaining elements
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}
#[cfg(target_arch = "x86_64")]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum: f32 = 0.0;
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        unsafe {
            let a_slice = _mm256_loadu_ps(a[i..i + 8].as_ptr());
            let b_slice = _mm256_loadu_ps(b[i..i + 8].as_ptr());
            let dot_prod = _mm256_dp_ps(a_slice, b_slice, 0xff);
            let tmp = _mm256_hadd_ps(dot_prod, dot_prod);
            sum += _mm256_cvtss_f32(tmp);
        }
        i += 8;
    }

    // Process remaining elements
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}
fn strassens(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = A.len();

    // BASE CASE: If the matrices are smaller than 256x256, use a nested loop to calculate
    // the dot product of the two matrices

    if n < 8 {
        let mut C = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        // let C = unpad(C);
        return C;
    }
    if n < 512 {
        let mut C = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                C[i][j] = dot_product(&A[i], &B[j]);
            }
        }
        // let C = unpad(C);
        return C;
    }
    // Step 2: Partition A, B, and C into n/2 x n/2 submatrices
    let (A11, A12, A21, A22, B11, B12, B21, B22) = partition(A, B);

    // Step 3: Define intermediate matrices
    let M1 = strassens(&add(&A11, &A22), &add(&B11, &B22));
    let M2 = strassens(&add(&A21, &A22), &B11);
    let M3 = strassens(&A11, &sub(&B12, &B22));
    let M4 = strassens(&A22, &sub(&B21, &B11));
    let M5 = strassens(&add(&A11, &A12), &B22);
    let M6 = strassens(&sub(&A21, &A11), &add(&B11, &B12));
    let M7 = strassens(&sub(&A12, &A22), &add(&B21, &B22));

    // Step 4: Compute the submatrices of the result matrix
    let C11 = add(&sub(&add(&M1, &M4), &M5), &M7);
    let C12 = add(&M3, &M5);
    let C21 = add(&M2, &M4);
    let C22 = add(&sub(&add(&M1, &M3), &M2), &M6);

    // Step 5: Return the combined submatrices
    let C = merge(&C11, &C12, &C21, &C22);

    C
}

fn strassens_matmul(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    // Step 1: Check if the matrices are of size 2^n x 2^n.
    // If not, pad the matrices with zeros to reach this size.
    let (A_padded, B_padded) = pad(A, B);

    let C = strassens(&A_padded, &B_padded);
    // Step 6: Unpad the matrix. Will do no-op unless necessary
    let C = unpad(C);

    C
}

// // Function to add two matrices
// fn add(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
//     let n = A.len();
//     let mut C = vec![vec![0.0; n]; n];
//     for i in 0..n {
//         for j in 0..n {
//             let a = A[i][j];
//             let b = B[i][j];
//             C[i][j] = a + b;
//         }
//     }
//     C
// }

#[cfg(target_arch = "aarch64")]
fn add(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    use std::arch::aarch64::*;

    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (0..n).step_by(4) {
            unsafe {
                let a = vld1q_f32(A[i][j..].as_ptr());
                let b = vld1q_f32(B[i][j..].as_ptr());
                let c = vaddq_f32(a, b);
                vst1q_f32(C[i][j..].as_mut_ptr(), c);
            }
        }
    }
    C
}

#[cfg(target_arch = "x86_64")]
fn add(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (0..n).step_by(8) {
            unsafe {
                let a = _mm256_loadu_ps(A[i][j..].as_ptr());
                let b = _mm256_loadu_ps(B[i][j..].as_ptr());
                let c = _mm256_add_ps(a, b);
                _mm256_storeu_ps(C[i][j..].as_mut_ptr(), c);
            }
        }
    }
    C
}

// // Function to subtract two matrices
// fn sub(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
//     let n = A.len();
//     let mut C = vec![vec![0.0; n]; n];
//     for i in 0..n {
//         for j in 0..n {
//             C[i][j] = A[i][j] - B[i][j];
//         }
//     }
//     C
// }

#[cfg(target_arch = "aarch64")]
fn sub(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    use std::arch::aarch64::*;

    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (0..n).step_by(4) {
            unsafe {
                let a = vld1q_f32(A[i][j..].as_ptr());
                let b = vld1q_f32(B[i][j..].as_ptr());
                let c = vsubq_f32(a, b);
                vst1q_f32(C[i][j..].as_mut_ptr(), c);
            }
        }
    }
    C
}

#[cfg(target_arch = "x86_64")]
fn sub(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (0..n).step_by(8) {
            unsafe {
                let a = _mm256_loadu_ps(A[i][j..].as_ptr());
                let b = _mm256_loadu_ps(B[i][j..].as_ptr());
                let c = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(C[i][j..].as_mut_ptr(), c);
            }
        }
    }
    C
}

// Function to combine two matrices into one
fn merge(
    C11: &Vec<Vec<f32>>,
    C12: &Vec<Vec<f32>>,
    C21: &Vec<Vec<f32>>,
    C22: &Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let n = C11.len();

    let mut C = vec![vec![0.0; n + n]; n + n];

    // Step 1: Combine the submatrices
    for i in 0..n {
        for j in 0..n {
            C[i][j] = C11[i][j];
            C[i][j + n] = C12[i][j];
            C[i + n][j] = C21[i][j];
            C[i + n][j + n] = C22[i][j];
        }
    }

    C
}
pub fn unpad(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut new_matrix = matrix.clone();

    // Find the last row and column that contain only zeroes
    if let Some((last_nonzero_row, last_nonzero_col)) = find_last_nonzero_row_col(&matrix) {
        new_matrix.truncate(last_nonzero_row);
        for row in &mut new_matrix {
            row.truncate(last_nonzero_col);
        }
    };

    new_matrix
}

fn find_last_nonzero_row_col(matrix: &Vec<Vec<f32>>) -> Option<(usize, usize)> {
    let mut ret = Option::None;
    let mut row = matrix.len();
    for i in (0..matrix.len()).rev() {
        if matrix[i].iter().all(|f| f.le(&1e-10f32)) {
            row -= 1;
        } else {
            break;
        }
    }
    let mut col = matrix[0].len();
    for i in (0..matrix[0].len()).rev() {
        if matrix.iter().all(|f| f[i].le(&1e-10f32)) {
            col -= 1;
        } else {
            break;
        }
    }

    if row != matrix.len() || col != matrix[0].len() {
        ret = Some((row, col))
    }

    ret
}
pub fn pad(matrix_1: &Vec<Vec<f32>>, matrix_2: &Vec<Vec<f32>>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = matrix_1.len();
    let m = matrix_1[0].len();
    let m2 = matrix_2.len();
    let p = matrix_2[0].len();
    assert_eq!(m, m2);

    let t = std::cmp::max(n, std::cmp::max(m, p));
    let q = t.next_power_of_two();

    let mut new_m1 = vec![vec![0.0; q]; q];
    let mut new_m2 = vec![vec![0.0; q]; q];

    for (i, row) in matrix_1.iter().enumerate() {
        for (j, item) in row.iter().enumerate() {
            new_m1[i][j] = *item;
        }
    }

    for (i, row) in matrix_2.iter().enumerate() {
        for (j, item) in row.iter().enumerate() {
            new_m2[i][j] = *item;
        }
    }

    (new_m1, new_m2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate(x: usize, y: usize) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(x);
        for _ in 0..x {
            let mut row = Vec::with_capacity(y);
            for _ in 0..y {
                row.push(1.);
            }
            result.push(row);
        }
        result
    }
    #[test]
    fn test_strassens_algorithm() {
        let A = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let B = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let expected_result = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        let result = strassens(&A, &B);
        assert_eq!(result, expected_result,);
    }
    #[test]
    fn test_strassens_algorithm_large() {
        let A = generate(1024, 1024);
        let B = generate(1024, 1024);
        let expected_result = generate(10, 10);
        let result = strassens(&A, &B);
        assert_eq!(result, expected_result,);
    }
    #[test]
    fn test_unpad_matrices() {
        let padded_matrix = vec![
            vec![1.0, 2.0, 0.0, 0.0],
            vec![3.0, 4.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        let unpadded_matrix = unpad(padded_matrix);

        let expected_matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        assert_eq!(unpadded_matrix, expected_matrix);
    }

    #[test]
    fn test_unpad_matrices_unchanged() {
        let padded_matrix = vec![
            vec![1.0, 2.0, 1.0, 1.0],
            vec![3.0, 4.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        let unpadded_matrix = unpad(padded_matrix);

        let expected_matrix = [
            [1.0, 2.0, 1.0, 1.0],
            [3.0, 4.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];

        assert_eq!(unpadded_matrix, expected_matrix);
    }
    #[test]
    fn test_pad_matrices() {
        let input_matrix_1 = vec![
            vec![5.0, 6.0, 7.0, 0.0],
            vec![5.0, 6.0, 7.0, 0.0],
            vec![5.0, 6.0, 7.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let input_matrix_2 = vec![
            vec![5.0, 6.0, 7.0, 0.0],
            vec![5.0, 6.0, 7.0, 0.0],
            vec![5.0, 6.0, 7.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        let (output_matrix_1, output_matrix_2) = pad(&input_matrix_1, &input_matrix_2);

        let expected_matrix_1 = [
            [5.0, 6.0, 7.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let expected_matrix_2 = [
            [5.0, 6.0, 7.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(output_matrix_1, expected_matrix_1);
        assert_eq!(output_matrix_2, expected_matrix_2);
    }
}
