pub(crate) fn partition(
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
            A12[i][j] = A[i][j + midpoint];
            A21[i][j] = A[i + midpoint][j];
            A22[i][j] = A[i + midpoint][j + midpoint];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + midpoint];
            B21[i][j] = B[i + midpoint][j];
            B22[i][j] = B[i + midpoint][j + midpoint];
        }
    }

    (A11, A12, A21, A22, B11, B12, B21, B22)
}
// Function to combine two matrices into one
pub(crate) fn merge(
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
