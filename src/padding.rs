pub(crate) fn pad(
    matrix_1: &Vec<Vec<f32>>,
    matrix_2: &Vec<Vec<f32>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
