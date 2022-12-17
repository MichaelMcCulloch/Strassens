pub(crate) fn add(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            C[i][j] = A[i][j] + B[i][j]
        }
    }
    C
}

pub(crate) fn sub(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = A.len();
    let mut C = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            C[i][j] = A[i][j] - B[i][j]
        }
    }
    C
}
