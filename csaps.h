#pragma once
#ifndef CSAPS_H
#define CSAPS_H

#include <vector>

#include <Eigen/Dense>


namespace csaps
{

typedef Eigen::VectorXd DoubleVector;


class UnivariateCubicSmoothingSpline
{
public:
  UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata);
  UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, const DoubleVector &weights);
  UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, double smooth);
  UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, const DoubleVector &weights, double smooth);

  DoubleVector operator()(const DoubleVector &xidata);
  DoubleVector operator()(size_t pcount);

protected:
  void MakeSpline();
  DoubleVector Evaluate(const DoubleVector &xidata);
  static DoubleVector Diff(const DoubleVector &vec);

protected:
  DoubleVector m_xdata;
  DoubleVector m_ydata;
  DoubleVector m_weights;

  double m_smooth;
};

} // namespace csaps

#endif // CSAPS_H
