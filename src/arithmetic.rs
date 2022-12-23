use crate::strassens::Matrix;

pub(crate) fn add(A: &Matrix, B: &Matrix) -> Matrix {
    match (A, B) {
        (Matrix::Zero, Matrix::Zero) => Matrix::Zero,
        (Matrix::Matrix(A), Matrix::Zero) => Matrix::Matrix(A.clone()),
        (Matrix::Zero, Matrix::Matrix(B)) => Matrix::Matrix(B.clone()),
        (Matrix::Matrix(A), Matrix::Matrix(B)) => {
            let n = A.len();
            let mut C = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    C[i][j] = A[i][j] + B[i][j]
                }
            }
            Matrix::Matrix(C)
        }
    }
}

pub(crate) fn sub(A: &Matrix, B: &Matrix) -> Matrix {
    match (A, B) {
        (Matrix::Zero, Matrix::Zero) => Matrix::Zero,
        (Matrix::Zero, Matrix::Matrix(B)) => {
            let n = B.len();
            let mut C = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    C[i][j] = 0.0 - B[i][j]
                }
            }

            Matrix::Matrix(C)
        }
        (Matrix::Matrix(A), Matrix::Zero) => Matrix::Matrix(A.clone()),
        (Matrix::Matrix(A), Matrix::Matrix(B)) => {
            let n = A.len();
            let mut C = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    C[i][j] = A[i][j] - B[i][j]
                }
            }

            Matrix::Matrix(C)
        }
    }
}
