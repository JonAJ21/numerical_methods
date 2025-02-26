#include <iostream>
#include <vector>
#include <iomanip>

#include "matrix.h"



int main() {
    Matrix A = Matrix({
        {-7, 3, -4, 7},
        {8, -1, -7, 6},
        {9, 9, 3, -6},
        {-7, -9, -8, -5}
    });
    std::cout << "Matrix A:\n" << A << std::endl;
    
    std::vector<double> b = {-126, 29, 27, 34};
    std::cout << "Vector b^T:\n";
    for (int i = 0; i < b.size(); i++) {
        std::cout << std::setw(10) << b[i] << " ";
    }
    std::cout << std::endl;

    std::vector<double> x = A.solveSLAE(b);
    std::cout << "Vector x^T:\n";
    for (int i = 0; i < x.size(); i++) {
        std::cout << std::setw(10) << x[i] << " ";
    }
    std::cout << std::endl;

    // Matrix A = Matrix(2,2);
    // std::cout << "Matrix A:\n" << A << std::endl;



    return 0;
}