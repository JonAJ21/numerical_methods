#include <vector>
#include <stdexcept>
#include <iomanip>

class Matrix {
private:
    int rows;
    int columns;
    std::vector<std::vector<double>> data;

public:
    Matrix(int rows, int columns, double initial = 0.0);
    Matrix(const std::vector<std::vector<double>>& list);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix() = default;

    int getRows() const;
    int getColumns() const;
    std::vector<std::vector<double>>& getData();

    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    Matrix transpose() const;
    std::pair<Matrix, Matrix> LUDecomposition() const;

    bool isTridiagonal() const;

    std::vector<double> solveSLAEWithLUDecompositionMethod(const std::vector<double>& b) const;
    std::vector<double> solveSLAEWithTridiagonalMethod(const std::vector<double>& d) const;

    Matrix inverse() const;
    double determinant() const;

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};
