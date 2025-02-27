#include <iostream>
#include <vector>
#include <iomanip>

#include "matrix.h"



int main() {
    Matrix A = Matrix({
        {-7, -9, 1, -9},
        {-6, -8, -5, 2},
        {-3, 6, 5, -9},
        {-2, 0, -5, -9}
    });
    std::cout << "Matrix A:\n" << A << std::endl;
    
    std::vector<double> b = {29, 42, 11, 75};
    std::cout << "Vector b^T:\n";
    for (int i = 0; i < b.size(); i++) {
        std::cout << std::setw(10) << b[i] << " ";
    }
    std::cout << std::endl;

    std::vector<double> x = A.solveSLAEWithLUDecompositionMethod(b);
    std::cout << "Vector x^T:\n";
    for (int i = 0; i < x.size(); i++) {
        std::cout << std::setw(10) << x[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Matrix A^-1:\n" << A.inverse() << std::endl;


    std::cout << "Matrix A * A^-1:\n" << A * A.inverse() << std::endl;


    std::cout << "Determinant of A: " << A.determinant() << std::endl;
    return 0;
}