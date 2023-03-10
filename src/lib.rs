mod arithmetic;
mod padding;
mod partition;
mod strassens;
mod unrolled_mult;
#[cfg(test)]
mod tests {
    use std::{
        ops::{Div, Sub},
        time::Instant,
    };

    use rustfft::num_complex::Complex32;

    use crate::strassens::strassens_matmul;

    use super::*;

    fn generate(x: usize, y: usize) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(x);
        for _ in 0..x {
            let mut row = Vec::with_capacity(y);
            for _ in 0..y {
                row.push(1.1);
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
        let result = strassens_matmul(&A, &B);
        assert_eq!(result, expected_result,);
    }
    #[test]
    fn test_strassens_algorithm_large() {
        let n = 4096 * 2 * 2;
        let A = generate(n, n);
        let B = generate(n, n);
        let result = strassens_matmul(&A, &B);

        println!("DONE STRASSENS");
        assert_eq!(result.len(), n,);
        assert_eq!(result[0].len(), n,);
        let mut C = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        assert_eq!(result.len(), n,);
        assert_eq!(result[0].len(), n,);
        assert_eq!(result, C)
    }
    #[test]
    fn test_strassens_algorithm_use_case() {
        let A = generate(3000, 201);
        let B = generate(201, 80);
        let start = Instant::now();
        let result = strassens_matmul(&A, &B);
        let end = Instant::now();

        println!("{}", end.sub(start).as_millis());
        println!("DONE STRASSENS");
        let mut C = vec![vec![0.0; 80]; 3000];
        let start = Instant::now();
        for i in 0..3000 {
            for j in 0..80 {
                for k in 0..201 {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        let end = Instant::now();
        println!("{}", end.sub(start).as_millis());
        assert_eq!(result.len(), 3000,);
        assert_eq!(result[0].len(), 80,);
        assert_eq!(result, C)
    }
}
