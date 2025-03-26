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
    Matrix A = *this; 

    std::vector<int> rowPermutations(n); 
    for (int i = 0; i < n; ++i) {
        rowPermutations[i] = i;
    }

    for (int i = 0; i < n; i++) {
        int maxRow = i;
        double maxVal = std::abs(A(i, i));
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A(k, i)) > maxVal) {
                maxVal = std::abs(A(k, i));
                maxRow = k;
            }
        }

        if (maxRow != i) {
            std::swap(rowPermutations[i], rowPermutations[maxRow]);
            for (int j = 0; j < n; j++) {
                std::swap(A(i, j), A(maxRow, j));
            }
        }

        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L(i, j) * U(j, k);
            }
            U(i, k) = A(i, k) - sum;
        }

        for (int k = i; k < n; k++) {
            if (i == k) {
                L(i, i) = 1.0;
            } else {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += L(k, j) * U(j, i);
                }
                L(k, i) = (A(k, i) - sum) / U(i, i);
            }
        }
    }

    return {L, U};
}

std::vector<double> Matrix::solveSLAEWithLUDecompositionMethod(const Matrix& L, const Matrix& U, const std::vector<double>& b) {
    int rows = L.getRows();
    int columns = L.getColumns();
    
    if (rows != columns) {
        throw std::invalid_argument("Matrix must be square");
    }
    if (b.size() != rows) {
        throw std::invalid_argument("Vector b must have the same length as the number of rows in the matrix");
    }

    int n = rows;
    std::vector<double> y(n), x(n);

    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L(i, j) * y[j];
        }
        y[i] /= L(i, i);
    }

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
    auto [L, U] = LUDecomposition();
    for (int i = 0; i < n; i++) {
        std::vector<double> b(n, 0.0);
        b[i] = 1.0; 
    
        auto x = solveSLAEWithLUDecompositionMethod(L, U, b);

        for (int j = 0; j < n; j++) {
            inv(j, i) = x[j];
        }
    }

    return inv;
} 

double Matrix::chebyshev_norm(const std::vector<double>& vec) {
    double norm = 0;
    for (double val : vec) {
        norm = std::max(norm, std::abs(val));
    }
    return norm;
}

double Matrix::mat_norm() const {
    double norm = -1e9; // -INFINITY
    for (const auto& row : data) {
        double row_sum = 0;
        for (double val : row) {
            row_sum += std::abs(val);
        }
        norm = std::max(norm, row_sum);
    }
    return norm;
}

double Matrix::line_rate_norm() const {
    double norm = 0;
    for (const auto& row : data) {
        double lineSum = 0;
        for (double val : row) {
            lineSum += std::abs(val);
        }
        norm = std::max(norm, lineSum);
    }
    return norm;
}

std::vector<double> Matrix::subtractVectors(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }

    std::vector<double> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }

    return result;
}

std::pair<std::vector<double>, int> Matrix::iterations(const std::vector<double>& b, double e) const {
    int n = rows;
    std::vector<std::vector<double>> A1(n, std::vector<double>(n, 0));
    std::vector<double> b1(n);
    std::vector<double> x(n);

    for (int i = 0; i < n; ++i) {
        b1[i] = b[i] / data[i][i];
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A1[i][j] = -data[i][j] / data[i][i];
            }
        }
    }

    double e1 = 1;
    x = b1;
    int num = 0;

    while (e1 > e) {
        std::vector<double> x1(n);

        for (int i = 0; i < n; ++i) {
            x1[i] = b1[i];
            for (int j = 0; j < n; ++j) {
                x1[i] += A1[i][j] * x[j];
            }
        }
        
        e1 = mat_norm() / (1 - Matrix(A1).mat_norm()) * chebyshev_norm(subtractVectors(x, x1));

        x = x1;
        num++;
    }


    return {x, num};
}


std::pair<std::vector<double>, int> Matrix::zeydel(const std::vector<double>& b, double e) const {
    int n = rows;
    std::vector<std::vector<double>> A1(n, std::vector<double>(n, 0));
    std::vector<double> b1(n, 0);
    std::vector<double> x0(n, 0);
    std::vector<double> x(n, 0);
    double eps_k = 1;

    for (int i = 0; i < n; ++i) {
        b1[i] = b[i] / data[i][i];
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A1[i][j] = -data[i][j] / data[i][i];
            }
        }
    }

    int num = 0;

    while (true) {
        num++;
        x0 = x;

        for (int i = 0; i < n; ++i) {
            x[i] = b1[i];
            for (int j = 0; j < n; ++j) {
                x[i] += A1[i][j] * x[j];
            }
        }

        eps_k = mat_norm() / (1 - Matrix(A1).mat_norm()) * chebyshev_norm(subtractVectors(x, x0));

        if (eps_k < e) {
            break;
        }
    }

    return {x, num};
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