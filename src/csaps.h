#pragma once
#ifndef CSAPS_H
#define CSAPS_H

#include <Eigen/Dense>


namespace csaps
{

typedef Eigen::ArrayXd DoubleArray;


class UnivariateCubicSmoothingSpline
{
public:
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth);
  UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth);

  DoubleArray operator()(const DoubleArray &xidata);
  DoubleArray operator()(size_t pcount, DoubleArray &xidata);

protected:
  typedef Eigen::ArrayXXd Coeffs;

  void MakeSpline();
  DoubleArray Evaluate(const DoubleArray &xidata);
  static DoubleArray Diff(const DoubleArray &vec);

protected:
  DoubleArray m_xdata;
  DoubleArray m_ydata;
  DoubleArray m_weights;

  double m_smooth;

  Coeffs m_coeffs;
};

} // namespace csaps

#endif // CSAPS_H
