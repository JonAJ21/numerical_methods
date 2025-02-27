#include "matrix.h"
#include <iostream>
Matrix::Matrix(int rows, int columns, double initial) {
    this->rows = rows;
    this->columns = columns;
    this->data = std::vector<std::vector<double>>(rows, std::vector<double>(columns, initial));
}

Matrix::Matrix(const std::vector<std::vector<double>>& list) {
    this->rows = list.size();
    this->columns = list[0].size();
    this->data.resize(this->rows, std::vector<double>(this->columns));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            this->data[i][j] = list[i][j];
        }
    }
}

Matrix::Matrix(const Matrix& other) {
    this->rows = other.rows;
    this->columns = other.columns;
    this->data.resize(this->rows, std::vector<double>(this->columns));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            this->data[i][j] = other.data[i][j];
        }
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
    this->rows = other.rows;
    this->columns = other.columns;
    
    this->data.resize(this->rows, std::vector<double>(this->columns));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            this->data[i][j] = other.data[i][j];
        }
    }
    
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept {
    this->rows = other.rows;
    this->columns = other.columns;
    this->data = std::move(other.data);
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    this->rows = other.rows;
    this->columns = other.columns;
    this->data = std::move(other.data);
    return *this;
}

int Matrix::getRows() const {
    return this->rows;
}

int Matrix::getColumns() const {
    return this->columns;
}

std::vector<std::vector<double>>& Matrix::getData() {
    return this->data;
}

double& Matrix::operator()(int row, int col) {
    if (row < 0 || row >= this->rows || col < 0 || col >= this->columns) {
        throw std::out_of_range("Index out of range");
    }
    
    return this->data[row][col];
}

double Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= this->rows || col < 0 || col >= this->columns) {
        throw std::out_of_range("Index out of range");
    }
    
    return this->data[row][col];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (this->rows != other.rows || this->columns != other.columns) {
        throw std::invalid_argument("Matrices must have the same dimensions");
    }
    
    Matrix result(this->rows, this->columns);
    
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            result(i, j) = this->data[i][j] + other.data[i][j];
        }
    }
    
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (this->columns != other.rows) {
        throw std::invalid_argument("Matrices must be multiplyable");
    }
    
    Matrix result(this->rows, other.columns);
    
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < other.columns; j++) {
            for (int k = 0; k < this->columns; k++) {
                result(i, j) += this->data[i][k] * other.data[k][j];
            }
        }
    }
    
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(this->rows, this->columns);
    
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            result(i, j) = this->data[i][j] * scalar;
        }
    }
    
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(this->columns, this->rows);
    
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            result(j, i) = this->data[i][j];
        }
    }
    
    return result;
}

bool Matrix::isTridiagonal() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (abs(i - j) > 1 && data[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}

std::pair<Matrix, Matrix> Matrix::LUDecomposition() const {
    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square");
    }
    int n = rows;
    Matrix L(n, n);
    Matrix U(n, n);

    for (int i = 0; i < n; i++) {
        // Upper triangular matrix U
        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L(i, j) * U(j, k);
            }
            U(i, k) = data[i][k] - sum;
        }

        // Lower triangular matrix L
        for (int k = i; k < n; k++) {
            if (i == k) {
                L(i, i) = 1.0;
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += L(k, j) * U(j, i);
                }
                L(k, i) = (data[k][i] - sum) / U(i, i);
            }
        }
    }
    return {L, U}; 
}

std::vector<double> Matrix::solveSLAEWithLUDecompositionMethod(const std::vector<double>& b) const {
    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square");
    }
    if (b.size() != rows) {
        throw std::invalid_argument("Vector b must have the same length as the number of rows in the matrix");
    }

    auto [L, U] = LUDecomposition();
    int n = rows;
    std::vector<double> y(n), x(n);

    // Direct substitution (Ly = b)
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L(i, j) * y[j];
        }
        y[i] /= L(i, i);
    }

    // Direct substitution (Ly = b)
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= U(i, j) * x[j];
        }
        x[i] /= U(i, i);
    }

    return x;
}

std::vector<double> Matrix::solveSLAEWithTridiagonalMethod(const std::vector<double>& d) const {
    if (rows == 0 || columns != rows) {
        throw std::invalid_argument("Matrix must be square");
    }

    if (!isTridiagonal()) {
        throw std::invalid_argument("Matrix must be tridiagonal");
    }

    
    std::vector<double> p(rows), q(rows);
    p[0] = - data[0][1] / data[0][0];
    q[0] = d[0] / data[0][0];

    for (int i = 1; i < rows - 1; i++) {
        p[i] = - data[i][i + 1] / (data[i][i] + data[i][i - 1] * p[i - 1]);
        q[i] = (d[i] - data[i][i - 1] * q[i - 1]) / (data[i][i] + data[i][i - 1] * p[i - 1]);
    }

    p[rows - 1] = 0.0;
    q[rows - 1] = (d[rows - 1] - data[rows - 1][rows - 2] * q[rows - 2]) / (data[rows - 1][rows - 1] + data[rows - 1][rows - 2] * p[rows - 2]);

    std::vector<double> x(rows);
    x[rows - 1] = q[rows - 1];
    for (int i = rows - 2; i >= 0; i--) {
        x[i] = q[i] + p[i] * x[i + 1];
    }

    return x;
}

double Matrix::determinant() const {
    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square");
    }

    auto [L, U] = LUDecomposition();
    double det = 1.0;
    for (int i = 0; i < rows; i++) {
        det *= U(i, i);
    }

    return det;
} 

Matrix Matrix::inverse() const {
    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square");
    }
    int n = rows;
    Matrix inv(n, n);

    // Solve the system for each column of the identity matrix
    for (int i = 0; i < n; i++) {
        std::vector<double> b(n, 0.0);
        b[i] = 1.0; // i-й столбец единичной матрицы

        auto x = solveSLAEWithLUDecompositionMethod(b);

        // Writing the solution in the inverse matrix
        for (int j = 0; j < n; j++) {
            inv(j, i) = x[j];
        }
    }

    return inv;
} 

std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            os << std::setw(10) << mat.data[i][j] << " ";
        }
        os << std::endl;
    }
    return os;
}
