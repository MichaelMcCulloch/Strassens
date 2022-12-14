#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64::*;
#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64::*;
fn split_matrix(
    matrix: &Vec<Vec<f32>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let A11 = matrix[0..rows / 2]
        .iter()
        .map(|row| row[0..cols / 2].to_vec())
        .collect();
    let A12 = matrix[0..rows / 2]
        .iter()
        .map(|row| row[cols / 2..cols].to_vec())
        .collect();
    let A21 = matrix[rows / 2..rows]
        .iter()
        .map(|row| row[0..cols / 2].to_vec())
        .collect();
    let A22 = matrix[rows / 2..rows]
        .iter()
        .map(|row| row[cols / 2..cols].to_vec())
        .collect();

    (A11, A12, A21, A22)
}
fn compute_dot_product(row1: &[f32], row2: &[f32]) -> f32 {
    let mut return_result = 0.0f32;
    let mut j = 0;

    // dot_product(&mut j, row1, row2, &mut return_result);

    // Calculate the dot product for the remaining elements
    if row1.len() != 4 {
        log::info!("")
    }
    for k in j..row1.len() {
        return_result += row1[k] * row2[k];
    }

    return_result
}

fn strassens(matrix1: &Vec<Vec<f32>>, matrix2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut result: Vec<Vec<f32>> = vec![vec![0f32; matrix2[0].len()]; matrix1.len()];
    // Create references to matrix1 and matrix2
    let (A11, A12, A21, A22) = split_matrix(&matrix1);
    let (B11, B12, B21, B22) = split_matrix(&matrix2);
    // Base case: Handle small matrices
    if matrix1.len() < 8 {
        let mut result: Vec<Vec<f32>> = vec![vec![0f32; matrix2[0].len()]; matrix1.len()];
        for i in 0..matrix1.len() {
            for j in 0..matrix2[0].len() {
                result[i][j] += compute_dot_product(&matrix1[i], &matrix2[j]);
            }
        }
        return result;
    }
    println!("Matrix Length: {} ", matrix1.len());

    // Create references to the submatrices
    let (A11, A12) = (&A11, &A12);
    let (A21, A22) = (&A21, &A22);
    let (B11, B12) = (&B11, &B12);
    let (B21, B22) = (&B21, &B22);

    let ((P1, P2, P3, P4), (P5, P6, P7, P8)) = (
        {
            let (P1, P2) = (strassens(A11, B11), strassens(A12, B21));
            let (P3, P4) = (strassens(A11, B12), strassens(A12, B22));
            (P1, P2, P3, P4)
        },
        {
            let (P5, P6) = (strassens(A21, B11), strassens(A22, B21));
            let (P7, P8) = (strassens(A21, B12), strassens(A22, B22));
            (P5, P6, P7, P8)
        },
    );

    result = sum_sub_matrices(&P1, &P2, &P3, &P4, &P5, &P6, &P7, &P8);

    if matrix1.len() == 16 {
        log::info!("fuck")
    }
    // Create references to the submatrices
    let S1 = subtract_matrices(&P3, &P5);
    let S2 = subtract_matrices(&P2, &P4);
    let S3 = subtract_matrices(&P1, &P7);
    let S4 = subtract_matrices(&P6, &P8);

    let C11 = add_matrices(&P1, &P4, &S3, &S2);
    let C12 = add_matrices(&P3, &P8, &S1, &S4);
    let C21 = add_matrices(&P2, &P8, &S1, &S2);
    let C22 = add_matrices(&P5, &P4, &S3, &S4);

    let mut result: Vec<Vec<f32>> = vec![vec![0f32; matrix2[0].len()]; matrix1.len()];
    let size = matrix1.len() / 2;
    for i in 0..size {
        for j in 0..size {
            result[i][j] = C11[i][j];
            result[i][j + size] = C12[i][j];
            result[i + size][j] = C21[i][j];
            result[i + size][j + size] = C22[i][j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_sub_matrices_simd() {
        let P1 = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ];
        let P2 = vec![
            vec![17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            vec![25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
        ];
        let P3 = vec![
            vec![33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
            vec![41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
        ];
        let P4 = vec![
            vec![49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0],
            vec![57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0],
        ];
        let P5 = vec![
            vec![65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0],
            vec![73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
        ];
        let P6 = vec![
            vec![81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0],
            vec![89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0],
        ];
        let P7 = vec![
            vec![97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            vec![105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0],
        ];
        let P8 = vec![
            vec![113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0],
            vec![121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0],
        ];

        let expected_result = vec![
            vec![456.0, 464.0, 472.0, 480.0, 488.0, 496.0, 504.0, 512.0],
            vec![520.0, 528.0, 536.0, 544.0, 552.0, 560.0, 568.0, 576.0],
        ];

        let result = unsafe { sum_sub_matrices(&P1, &P2, &P3, &P4, &P5, &P6, &P7, &P8) };
        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_sum_sub_matrices_simd_ones() {
        let P1 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P2 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P3 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P4 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P5 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P6 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P7 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let P8 = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];

        let expected_result = vec![
            vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        ];

        let result = unsafe { sum_sub_matrices(&P1, &P2, &P3, &P4, &P5, &P6, &P7, &P8) };
        assert_eq!(expected_result, result);
    }
    #[test]
    fn test_matrix_multiplication_simd() {
        let matrix1 = create_matrix(16, 16);
        let matrix2 = create_matrix(16, 16);

        let result = strassens(&matrix1, &matrix2);

        assert_eq!(
            result,
            vec![
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            ]
        );
    }

    fn create_matrix(rows: usize, columns: usize) -> Vec<Vec<f32>> {
        let mut matrix = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::with_capacity(columns);
            for _ in 0..columns {
                row.push(1.0);
            }
            matrix.push(row);
        }
        matrix
    }
}
