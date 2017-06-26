#pragma once
#ifndef CSAPS_H
#define CSAPS_H

#include <limits>

#include <Eigen/Dense>


namespace csaps
{

typedef Eigen::ArrayXd DoubleArray;
typedef Eigen::Array<Eigen::DenseIndex, Eigen::Dynamic, 1> IndexArray;

typedef std::numeric_limits<double> DoubleLimits;


class UnivariateCubicSmoothingSpline
{
public:
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth);

  DoubleArray operator()(const DoubleArray &xidata);
  DoubleArray operator()(const size_t pcount, DoubleArray &xidata);

  typedef Eigen::ArrayXXd CoeffsMatrix;

  double GetSmooth() const { return m_smooth; }
  const DoubleArray& GetBreaks() const { return m_xdata; }
  const CoeffsMatrix& GetCoeffs() const { return m_coeffs; }
  typename CoeffsMatrix::Index GetPieces() const { return m_coeffs.rows(); }

protected:

  void MakeSpline();
  DoubleArray Evaluate(const DoubleArray &xidata);

  //! Calculate the 1-th discrete difference
  static DoubleArray Diff(const DoubleArray &vec);

  //! Return the indices of the bins to which each value in input array belongs
  static IndexArray Digitize(const DoubleArray &arr, const DoubleArray &bins);

protected:
  DoubleArray m_xdata;
  DoubleArray m_ydata;
  DoubleArray m_weights;

  double m_smooth;

  CoeffsMatrix m_coeffs;
};

} // namespace csaps

#endif // CSAPS_H
