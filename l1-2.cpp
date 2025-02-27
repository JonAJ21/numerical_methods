#include <iostream>
#include <iomanip>
#include "matrix.h"

int main() {

    Matrix A({
        { 8, -4,  0,  0,  0},
        {-2, 12, -7,  0,  0},
        { 0,  2, -9,  1,  0},
        { 0,  0, -8, 17, -4},
        { 0,  0,  0, -7, 13}
    });

    std::cout << "Matrix A:\n" << A << std::endl;

    std::vector<double> d({32, 15, -10, 133, -76});

    std::cout << "Vector d^T:" << std::endl;

    for (double num : d) {
        std::cout << std::setw(10) << num << ' ';
    }
    std::cout << std::endl;

    auto x = A.solveSLAEWithTridiagonalMethod(d);
    std::cout << "Vector x^T (Ax=d):" << std::endl;

    for (double num : x) {
        std::cout << std::setw(10) << num << ' ';
    }
    std::cout << std::endl;

    std::cout << "Test:" << std::endl;

    for (int i = 0; i < A.getRows(); i++) {
        double res = 0;
        for (int j = 0; j < A.getColumns(); j++) {
            res += A(i, j) * x[j];
        }
        std::cout << "( " << res << " == " << d[i] << " ) = " << (((abs(res - d[i]) < 1e-9) == 1) ? "True" : "False") << std::endl;  
    }

    return 0;
}