#pragma once
#ifndef CSAPS_H
#define CSAPS_H

#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>


namespace csaps
{

typedef Eigen::DenseIndex Index;
typedef Index Size;
typedef Eigen::ArrayXd DoubleArray;
typedef Eigen::ArrayXXd DoubleArray2D;
typedef Eigen::Array<Index, Eigen::Dynamic, 1> IndexArray;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, Index> DoubleSparseMatrix;

typedef std::numeric_limits<double> DoubleLimits;


//! Calculates the 1-th discrete difference
DoubleArray Diff(const DoubleArray &vec);


//! Returns the indices of the bins to which each value in input array belongs
IndexArray Digitize(const DoubleArray &arr, const DoubleArray &bins);


//! Makes rows x cols sparse matrix from diagonals and offsets
DoubleSparseMatrix MakeSparseDiagMatrix(const DoubleArray2D& diags, const IndexArray& offsets, Size rows, Size cols);


//! Solves sparse linear system Ab = x via supernodal LU factorization
DoubleArray SolveLinearSystem(const DoubleSparseMatrix &A, const DoubleArray &b);


class UnivariateCubicSmoothingSpline
{
public:
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth);

  DoubleArray operator()(const DoubleArray &xidata);
  DoubleArray operator()(const Size pcount, DoubleArray &xidata);

  double GetSmooth() const { return m_smooth; }
  const DoubleArray& GetBreaks() const { return m_xdata; }
  const DoubleArray2D& GetCoeffs() const { return m_coeffs; }
  DoubleArray2D::Index GetPieces() const { return m_coeffs.rows(); }

protected:

  void MakeSpline();
  DoubleArray Evaluate(const DoubleArray &xidata);

protected:
  DoubleArray m_xdata;
  DoubleArray m_ydata;
  DoubleArray m_weights;

  double m_smooth;

  DoubleArray2D m_coeffs;
};

} // namespace csaps

#endif // CSAPS_H
