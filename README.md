# Strassens
Efficient implementation of matrix multiply


This is a Rust implementation of Strassen's matrix multiplication algorithm. It takes two matrices, A and B, and returns a matrix C which is the product of A and B. The algorithm works by partitioning the matrices into four submatrices, then computing seven intermediate matrices (M1-M7) using the submatrices. Finally, the intermediate matrices are combined to form the final matrix C. The algorithm is optimized for parallelism, using the Rayon library to take advantage of multiple cores.
