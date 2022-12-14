use std::cmp;

fn strassens_base_case(A: &mut Vec<Vec<f32>>, B: &mut Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let x = A.len();
    let y = A[0].len();
    let z = B[0].len();

    if x < 8 || y < 8 || z < 8 {
        let mut x = A.len();
        let mut y = A[0].len();
        let mut z = B[0].len();
        let mut C = vec![vec![0.0; z]; x];

        for i in 0..x {
            for j in 0..z {
                for k in 0..y {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    return vec![];
}
fn multiply_matrix(A: &mut Vec<Vec<f32>>, B: &mut Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut x = A.len();
    let mut y = A[0].len();
    let mut z = B[0].len();
    let mut C = vec![vec![0.0; z]; x];

    for i in 0..x {
        for j in 0..z {
            for k in 0..y {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}
fn strassens(A: &mut Vec<Vec<f32>>, B: &mut Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut c = strassens_base_case(A, B);
    if !c.is_empty() {
        return c;
    }
    let x = A.len();
    let y = A[0].len();
    let z = B[0].len();

    // pad A and B with zeroes if they have an odd number of dimensions
    if x % 2 != 0 {
        A.push(vec![0.0; y]);
    }
    if y % 2 != 0 {
        for row in A.iter_mut() {
            row.push(0.0);
        }
        B.push(vec![0.0; y]);
    }
    if z % 2 != 0 {
        for row in B.iter_mut() {
            row.push(0.0);
        }
    }

    let x1 = A.len();
    let y1 = A[0].len();
    let z1 = B[0].len();

    let mut a11 = split(A, 0, x1 / 2, 0, y1 / 2);
    let a12 = split(A, 0, x1 / 2, y1 / 2, y1);
    let a21 = split(A, x1 / 2, x1, 0, y1 / 2);
    let mut a22 = split(A, x1 / 2, x1, y1 / 2, y1);
    let mut b11 = split(B, 0, y1 / 2, 0, z1 / 2);
    let b12 = split(B, 0, y1 / 2, z1 / 2, z1);
    let b21 = split(B, y1 / 2, y1, 0, z1 / 2);
    let mut b22 = split(B, y1 / 2, y1, z1 / 2, z1);

    let m1 = strassens(&mut add(&a11, &a22), &mut add(&b11, &b22));
    let m2 = strassens(&mut add(&a21, &a22), &mut b11);
    let m3 = strassens(&mut a11, &mut sub(&b12, &b22));
    let m4 = strassens(&mut a22, &mut sub(&b21, &b11));
    let m5 = strassens(&mut add(&a11, &a12), &mut b22);
    let m6 = strassens(&mut sub(&a21, &a11), &mut add(&b11, &b12));
    let m7 = strassens(&mut sub(&a12, &a22), &mut add(&b21, &b22));

    let c11 = add(&sub(&add(&m1, &m4), &m5), &m7);
    let c12 = add(&m3, &m5);
    let c21 = add(&m2, &m4);
    let c22 = add(&sub(&add(&m1, &m3), &m2), &m6);

    let mut c = merge(&c11, &c12, &c21, &c22);

    // discard extra rows and columns, if necessary
    if x % 2 != 0 {
        c.pop();
    }
    if y % 2 != 0 {
        for row in c.iter_mut() {
            row.pop();
        }
    }

    return c;
}

fn merge(
    a11: &Vec<Vec<f32>>,
    a12: &Vec<Vec<f32>>,
    a21: &Vec<Vec<f32>>,
    a22: &Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut c = vec![vec![0.0; a11[0].len() + a12[0].len()]; a11.len() + a21.len()];

    for i in 0..a11.len() {
        for j in 0..a11[0].len() {
            c[i][j] = a11[i][j];
        }
    }

    for i in 0..a12.len() {
        for j in 0..a12[0].len() {
            c[i][j + a11[0].len()] = a12[i][j];
        }
    }

    for i in 0..a21.len() {
        for j in 0..a21[0].len() {
            c[i + a11.len()][j] = a21[i][j];
        }
    }

    for i in 0..a22.len() {
        for j in 0..a22[0].len() {
            c[i + a11.len()][j + a11[0].len()] = a22[i][j];
        }
    }

    return c;
}
fn split(
    A: &mut Vec<Vec<f32>>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> Vec<Vec<f32>> {
    let mut sub_matrix = vec![];

    for row in A[row_start..row_end].iter() {
        let mut sub_row = vec![];
        for col in row[col_start..col_end].iter() {
            sub_row.push(*col);
        }
        sub_matrix.push(sub_row);
    }

    return sub_matrix;
}
fn add(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let x = cmp::min(a.len(), b.len());
    let y = cmp::min(a[0].len(), b[0].len());
    let mut c = vec![vec![0.0; y]; x];

    for i in 0..x {
        for j in 0..y {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    return c;
}
fn sub(A: &Vec<Vec<f32>>, B: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let x = cmp::min(A.len(), B.len());
    let y = cmp::min(A[0].len(), B[0].len());
    let mut C = vec![vec![0.0; y]; x];

    for i in 0..x {
        for j in 0..y {
            C[i][j] = A[i][j] - B[i][j];
        }
    }

    return C;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_matrix1() {
        let mut matrix = vec![
            vec![1.0, 2.0, 3.0, 9.0],
            vec![4.0, 5.0, 6.0, 10.0],
            vec![21.0, 22.0, 23.0, 24.0],
            vec![7.0, 8.0, 9.0, 11.0],
        ];

        // Test the splitting of the top left quadrant
        let top_left_quadrant = split(&mut matrix, 0, 2, 0, 2);
        assert_eq!(top_left_quadrant, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);

        // Test the splitting of the top right quadrant
        let top_right_quadrant = split(&mut matrix, 0, 2, 2, 4);
        assert_eq!(top_right_quadrant, vec![vec![3.0, 9.0], vec![6.0, 10.0]]);

        // Test the splitting of the bottom left quadrant
        let bottom_left_quadrant = split(&mut matrix, 2, 4, 0, 2);
        assert_eq!(bottom_left_quadrant, vec![vec![21.0, 22.0], vec![7.0, 8.0]]);

        // Test the splitting of the bottom right quadrant
        let bottom_right_quadrant = split(&mut matrix, 2, 4, 2, 4);
        assert_eq!(
            bottom_right_quadrant,
            vec![vec![23.0, 24.0], vec![9.0, 11.0]]
        );
    }
    #[test]
    fn test_combine_matrix() {
        let a11 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let a12 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let a21 = vec![vec![9.0, 10.0], vec![11.0, 12.0]];
        let a22 = vec![vec![13.0, 14.0], vec![15.0, 16.0]];

        let c = merge(&a11, &a12, &a21, &a22);

        assert_eq!(
            c,
            vec![
                vec![1.0, 2.0, 5.0, 6.0],
                vec![3.0, 4.0, 7.0, 8.0],
                vec![9.0, 10.0, 13.0, 14.0],
                vec![11.0, 12.0, 15.0, 16.0]
            ]
        );
    }
    #[test]
    #[test]
    fn test_multiply_matrix() {
        let mut A = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mut B = vec![vec![2.0, 0.0], vec![1.0, 2.0]];

        let C = multiply_matrix(&mut A, &mut B);

        assert_eq!(C, vec![vec![4.0, 4.0], vec![10.0, 8.0]]);
    }
    #[test]
    fn test_matrix_multiplication() {
        let mut matrix1 = create_matrix(17, 17);
        let mut matrix2 = create_matrix(17, 17);

        let result = strassens(&mut matrix1, &mut matrix2);

        assert_eq!(
            result,
            matrix_multiply(&create_matrix(17, 17), &create_matrix(17, 17))
        );
    }
    fn matrix_multiply(matrix_a: &Vec<Vec<f32>>, matrix_b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut matrix_c = vec![vec![0.0; matrix_b[0].len()]; matrix_a.len()];

        for row in 0..matrix_a.len() {
            for col in 0..matrix_b[0].len() {
                for k in 0..matrix_a[0].len() {
                    matrix_c[row][col] += matrix_a[row][k] * matrix_b[k][col];
                }
            }
        }
        matrix_c
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
