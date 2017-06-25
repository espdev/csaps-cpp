#include <iostream>

#include "csaps.h"

namespace csaps
{

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights)
  : UnivariateCubicSmoothingSpline(xdata, ydata, weights, -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), smooth)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth)
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

DoubleArray UnivariateCubicSmoothingSpline::operator()(const DoubleArray &xidata)
{
  if (xidata.size() < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  return Evaluate(xidata);
}

DoubleArray UnivariateCubicSmoothingSpline::operator()(size_t pcount)
{
  if (pcount < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  DoubleArray xidata = DoubleArray::LinSpaced(pcount, m_xdata(0), m_xdata(m_xdata.size()-1));

  return Evaluate(xidata);
}

void UnivariateCubicSmoothingSpline::MakeSpline()
{
  size_t pcount = m_xdata.size();

  auto dx = Diff(m_xdata);
  auto dy = Diff(m_ydata);
  auto divdxdy = dy / dx;

  if (pcount > 2) {

  }
  else {
    double p = 1.0;
    m_coeffs = Coeffs(1, 2);
    m_coeffs(0, 0) = divdxdy(0);
    m_coeffs(0, 1) = m_ydata(0);
    std::cout << m_coeffs << std::endl;
  }
}

DoubleArray UnivariateCubicSmoothingSpline::Evaluate(const DoubleArray & xidata)
{
  return DoubleArray();
}

DoubleArray UnivariateCubicSmoothingSpline::Diff(const DoubleArray &vec)
{
  size_t n = vec.size() - 1;
  return vec.tail(n) - vec.head(n);
}

} // namespace csaps
