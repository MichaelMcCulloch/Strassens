use crate::strassens::Matrix;

pub(crate) fn partition(
    A: &Vec<Vec<f32>>,
    B: &Vec<Vec<f32>>,
) -> (
    Matrix,
    Matrix,
    Matrix,
    Matrix,
    Matrix,
    Matrix,
    Matrix,
    Matrix,
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

    let mut A11_all_zeroes = true;
    let mut A12_all_zeroes = true;
    let mut A21_all_zeroes = true;
    let mut A22_all_zeroes = true;
    let mut B11_all_zeroes = true;
    let mut B12_all_zeroes = true;
    let mut B21_all_zeroes = true;
    let mut B22_all_zeroes = true;

    for i in 0..midpoint {
        for j in 0..midpoint {
            A11[i][j] = A[i][j];
            if !A11[i][j].eq(&0.0f32) {
                A11_all_zeroes = false;
            }
            A12[i][j] = A[i][j + midpoint];
            if !A12[i][j].eq(&0.0f32) {
                A12_all_zeroes = false;
            }
            A21[i][j] = A[i + midpoint][j];
            if !A21[i][j].eq(&0.0f32) {
                A21_all_zeroes = false;
            }
            A22[i][j] = A[i + midpoint][j + midpoint];
            if !A22[i][j].eq(&0.0f32) {
                A22_all_zeroes = false;
            }

            B11[i][j] = B[i][j];
            if !B11[i][j].eq(&0.0f32) {
                B11_all_zeroes = false;
            }
            B12[i][j] = B[i][j + midpoint];
            if !B12[i][j].eq(&0.0f32) {
                B12_all_zeroes = false;
            }
            B21[i][j] = B[i + midpoint][j];
            if !B21[i][j].eq(&0.0f32) {
                B21_all_zeroes = false;
            }
            B22[i][j] = B[i + midpoint][j + midpoint];
            if !B22[i][j].eq(&0.0f32) {
                B22_all_zeroes = false;
            }
        }
    }

    (
        if A11_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(A11)
        },
        if A12_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(A12)
        },
        if A21_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(A21)
        },
        if A22_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(A22)
        },
        if B11_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(B11)
        },
        if B12_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(B12)
        },
        if B21_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(B21)
        },
        if B22_all_zeroes {
            Matrix::Zero
        } else {
            Matrix::Matrix(B22)
        },
    )
}
// Function to combine two matrices into one
pub(crate) fn merge(C11: &Matrix, C12: &Matrix, C21: &Matrix, C22: &Matrix) -> Matrix {
    match (C11, C12, C21, C22) {
        (Matrix::Matrix(A), _, _, _)
        | (Matrix::Zero, Matrix::Matrix(A), _, _)
        | (Matrix::Zero, Matrix::Zero, Matrix::Matrix(A), _)
        | (Matrix::Zero, Matrix::Zero, Matrix::Zero, Matrix::Matrix(A)) => {
            let n = A.len();
            let mut C = vec![vec![0.0; n + n]; n + n];

            // Step 1: Combine the submatrices
            for i in 0..n {
                for j in 0..n {
                    C[i][j] = match C11 {
                        Matrix::Zero => 0.0f32,
                        Matrix::Matrix(C11) => C11[i][j],
                    };
                    C[i][j + n] = match C12 {
                        Matrix::Zero => 0.0f32,
                        Matrix::Matrix(C12) => C12[i][j],
                    };
                    C[i + n][j] = match C21 {
                        Matrix::Zero => 0.0f32,
                        Matrix::Matrix(C21) => C21[i][j],
                    };
                    C[i + n][j + n] = match C22 {
                        Matrix::Zero => 0.0f32,
                        Matrix::Matrix(C22) => C22[i][j],
                    };
                }
            }

            Matrix::Matrix(C)
        }
        (Matrix::Zero, Matrix::Zero, Matrix::Zero, Matrix::Zero) => Matrix::Zero,
    }
}
