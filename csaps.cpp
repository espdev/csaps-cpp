#include <iostream>

#include "csaps.h"

namespace csaps
{

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleVector(), -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, const DoubleVector &weights)
  : UnivariateCubicSmoothingSpline(xdata, ydata, weights, -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, double smooth)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleVector(), smooth)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleVector &xdata, const DoubleVector &ydata, const DoubleVector &weights, double smooth)
  : m_xdata(xdata)
  , m_ydata(ydata)
  , m_weights(weights)
  , m_smooth(smooth)
{
  if (m_xdata.size() < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  if (m_weights.size() == 0) {
    m_weights = Eigen::VectorXd::Constant(m_xdata.size(), 1.0);
  }

  if (m_smooth > 1.0) {
    throw std::exception("Smoothing parameter must be less than or equal 1.0");
  }

  if (m_xdata.size() != m_ydata.size() || m_xdata.size() != m_weights.size()) {
    throw std::exception("Lenghts of the input data vectors are not equal");
  }

  MakeSpline();
}

DoubleVector UnivariateCubicSmoothingSpline::operator()(const DoubleVector &xidata)
{
  if (xidata.size() < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  return Evaluate(xidata);
}

DoubleVector UnivariateCubicSmoothingSpline::operator()(size_t pcount)
{
  if (pcount < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  DoubleVector xidata = DoubleVector::LinSpaced(pcount, m_xdata(0), m_xdata(m_xdata.size()-1));

  return Evaluate(xidata);
}

void UnivariateCubicSmoothingSpline::MakeSpline()
{
  size_t pcount = m_xdata.size();
}

DoubleVector UnivariateCubicSmoothingSpline::Evaluate(const DoubleVector & xidata)
{
  return DoubleVector();
}

DoubleVector UnivariateCubicSmoothingSpline::Diff(const DoubleVector &vec)
{
  size_t n = vec.size() - 1;
  return vec.tail(n) - vec.head(n);
}

} // namespace csaps
