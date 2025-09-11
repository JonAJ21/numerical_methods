#include <iostream>
#include <iomanip>
#include "matrix.h"

int main() {

    Matrix A = Matrix({
        { 12, -3, -1, 3},
        { 5, 20, 9, 1},
        { 6, -3, -21, -7},
        { 0,  0, -8, 17, -4}
    });

    std::cout << "Matrix A:\n" << A << std::endl;

    std::vector<double> b = {-31, 90, 119, 71};
    std::cout << "Vector b^T:\n";
    for (int i = 0; i < b.size(); i++) {
        std::cout << std::setw(10) << b[i] << " ";
    }
    std::cout << std::endl;

    auto x1 = A.iterations(b, 0.0001);

    std::cout << "Vector iterations x^T:\n";
    for (int i = 0; i < x1.first.size(); i++) {
        std::cout << std::setw(10) << x1.first[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Number of iterations: " << x1.second << std::endl;

    auto x2 = A.zeydel(b, 0.0001);

    std::cout << "Vector zeydel x^T:\n";
    for (int i = 0; i < x2.first.size(); i++) {
        std::cout << std::setw(10) << x2.first[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Number of iterations: " << x2.second << std::endl;

    std::cout << "Test iterations:" << std::endl;

    for (int i = 0; i < A.getRows(); i++) {
        double res = 0;
        for (int j = 0; j < A.getColumns(); j++) {
            res += A(i, j) * x1.first[j];
        }
        std::cout << "( " << res << " == " << b[i] << " ) = " << (((abs(res - b[i]) < 0.0001) == 1) ? "True" : "False") << std::endl;
    }


    std::cout << "Test zeydel:" << std::endl;

    for (int i = 0; i < A.getRows(); i++) {
        double res = 0;
        for (int j = 0; j < A.getColumns(); j++) {
            res += A(i, j) * x2.first[j];
        }
        std::cout << "( " << res << " == " << b[i] << " ) = " << (((abs(res - b[i]) < 0.0001) == 1) ? "True" : "False") << std::endl;
    }


    return 0;
}