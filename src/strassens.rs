use crate::{
    arithmetic::{add, sub},
    padding::pad,
    partition::{merge, partition},
    unrolled_mult::long_mult,
};

pub enum Matrix {
    Zero,
    Matrix(Vec<Vec<f32>>),
}

pub(crate) fn strassens_matmul(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    // Step 1: Check if the matrices are of size 2^n x 2^n.
    // If not, pad the matrices with zeros to reach this size.

    let dim_M = A.len();
    let dim_n = B[0].len();

    let (A_padded, B_padded) = pad(A, B);

    let a_padded_len = A_padded.len();

    match strassens(&Matrix::Matrix(A_padded), &Matrix::Matrix(B_padded)) {
        Matrix::Zero => vec![vec![0.0; B[0].len()]; A.len()],
        Matrix::Matrix(mut C) => {
            // Step 6: Unpad the matrix. Will do no-op unless necessary
            assert!(C.len() == a_padded_len);
            assert!(C[0].len() == a_padded_len);
            C.truncate(dim_M);
            for i in C.iter_mut() {
                i.truncate(dim_n)
            }

            C
        }
    }
}

pub(crate) fn strassens(A: &Matrix, B: &Matrix) -> Matrix {
    match (A, B) {
        (Matrix::Zero, Matrix::Zero)
        | (Matrix::Matrix(_), Matrix::Zero)
        | (Matrix::Zero, Matrix::Matrix(_)) => Matrix::Zero,
        (Matrix::Matrix(A), Matrix::Matrix(B)) => {
            let n = A.len();

            if n < 8 {
                let mut C = vec![vec![0.0; 512]; 512];
                for i in 0..512 {
                    for j in 0..512 {
                        for k in 0..512 {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
                return Matrix::Matrix(C);
            }

            if n <= 512 {
                return Matrix::Matrix(long_mult(A, B));
            }

            let (A11, A12, A21, A22, B11, B12, B21, B22) = partition(A, B);

            let (M1, (M2, (M3, (M4, (M5, (M6, (M7))))))) = rayon::join(
                || strassens(&add(&A11, &A22), &add(&B11, &B22)),
                || {
                    rayon::join(
                        || strassens(&add(&A21, &A22), &B11),
                        || {
                            rayon::join(
                                || strassens(&A11, &sub(&B12, &B22)),
                                || {
                                    rayon::join(
                                        || strassens(&A22, &sub(&B21, &B11)),
                                        || {
                                            rayon::join(
                                                || strassens(&add(&A11, &A12), &B22),
                                                || {
                                                    rayon::join(
                                                        || {
                                                            strassens(
                                                                &sub(&A21, &A11),
                                                                &add(&B11, &B12),
                                                            )
                                                        },
                                                        || {
                                                            strassens(
                                                                &sub(&A12, &A22),
                                                                &add(&B21, &B22),
                                                            )
                                                        },
                                                    )
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        },
                    )
                },
            );

            let C11 = add(&sub(&add(&M1, &M4), &M5), &M7);
            let C12 = add(&M3, &M5);
            let C21 = add(&M2, &M4);
            let C22 = add(&sub(&add(&M1, &M3), &M2), &M6);

            // Step 5: Return the combined submatrices
            let C = merge(&C11, &C12, &C21, &C22);

            C
        }
    }
}
