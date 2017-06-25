#include <iostream>
#include <cmath>

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

DoubleArray UnivariateCubicSmoothingSpline::operator()(size_t pcount, DoubleArray &xidata)
{
  if (pcount < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  xidata = DoubleArray::LinSpaced(pcount, m_xdata(0), m_xdata(m_xdata.size()-1));

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
  }
}

DoubleArray UnivariateCubicSmoothingSpline::Evaluate(const DoubleArray & xidata)
{
  const auto x_size = m_xdata.size();

  auto mesh = m_xdata.segment(1, x_size - 2);
  DoubleArray edges(x_size);

  edges(0) = -DoubleLimits::infinity();
  edges.segment(1, x_size - 2) = mesh;
  edges(x_size - 1) = DoubleLimits::infinity();

  auto indexes = Digitize(xidata, edges);
  indexes -= 1;

  auto xi_size = xidata.size();

  DoubleArray xidata_loc(xi_size);
  DoubleArray yidata(xi_size);

  for (Eigen::DenseIndex i = 0; i < xi_size; ++i) {
    Eigen::DenseIndex index = indexes(i);

    // Go to local coordinates
    xidata_loc(i) = xidata(i) - m_xdata(index);

    // Initial values
    yidata(i) = m_coeffs(index, 0);
  }

  DoubleArray coeffs(xi_size);

  for (Eigen::DenseIndex i = 1; i < m_coeffs.cols(); ++i) {
    for (Eigen::DenseIndex k = 0; k < xi_size; ++k) {
      coeffs(k) = m_coeffs(indexes(k), i);
    }

    yidata = xidata_loc * yidata + coeffs;
  }

  return yidata;
}

DoubleArray UnivariateCubicSmoothingSpline::Diff(const DoubleArray &vec)
{
  size_t n = vec.size() - 1;
  return vec.tail(n) - vec.head(n);
}

IndexArray UnivariateCubicSmoothingSpline::Digitize(const DoubleArray &arr, const DoubleArray &bins)
{
  // This code works if `arr` and `bins` are monotonically increasing

  IndexArray indexes = IndexArray::Zero(arr.size());

  auto is_inside_bin = [arr, bins](Eigen::DenseIndex item, Eigen::DenseIndex index)
  {
    const double prc = 1.e-8;

    double a = arr(item);
    double bl = bins(index - 1);
    double br = bins(index);

    // bins[i-1] <= a < bins[i]
    return (a > bl || std::abs(a - bl) < std::abs(std::min(a, bl)) * prc) && a < br;
  };

  Eigen::DenseIndex kstart = 1;

  for (Eigen::DenseIndex i = 0; i < arr.size(); ++i) {
    for (Eigen::DenseIndex k = kstart; k < bins.size(); ++k) {
      if (is_inside_bin(i, k)) {
        indexes(i) = k;
        kstart = k;
        break;
      }
    }
  }

  return indexes;
}

} // namespace csaps
