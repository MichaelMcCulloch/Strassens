use crate::{
    arithmetic::{add, sub},
    padding::pad,
    partition::{merge, partition},
};

pub(crate) fn strassens_matmul(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    // Step 1: Check if the matrices are of size 2^n x 2^n.
    // If not, pad the matrices with zeros to reach this size.
    let (A_padded, B_padded) = pad(A, B);

    let mut C = strassens(&A_padded, &B_padded);
    // Step 6: Unpad the matrix. Will do no-op unless necessary
    assert!(C.len() == A_padded.len());
    assert!(C[0].len() == A_padded.len());
    C.truncate(A.len());
    for i in C.iter_mut() {
        i.truncate(B[0].len())
    }

    C
}

pub(crate) fn strassens(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = A.len();
    println!("{}x{}", n, n);

    // BASE CASE: If the matrices are smaller than 256x256, use a nested loop to calculate
    // the dot product of the two matrices

    if n == 512 {
        let mut C = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // Step 2: Partition A, B, and C into n/2 x n/2 submatrices
    let (A11, A12, A21, A22, B11, B12, B21, B22) = partition(A, B);

    // Step 3: Define intermediate matrices

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
                                                || strassens(&sub(&A21, &A11), &add(&B11, &B12)),
                                                || strassens(&sub(&A12, &A22), &add(&B21, &B22)),
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

    // Step 4: Compute the submatrices of the result matrix
    let C11 = add(&sub(&add(&M1, &M4), &M5), &M7);
    let C12 = add(&M3, &M5);
    let C21 = add(&M2, &M4);
    let C22 = add(&sub(&add(&M1, &M3), &M2), &M6);

    // Step 5: Return the combined submatrices
    let C = merge(&C11, &C12, &C21, &C22);

    C
}
