#include <cmath>
#include <algorithm>
#include <iostream>

#include "csaps.h"

namespace csaps
{

DoubleArray Diff(const DoubleArray &vec)
{
  const auto n = vec.size() - 1;
  return vec.tail(n) - vec.head(n);
}

IndexArray Digitize(const DoubleArray &arr, const DoubleArray &bins)
{
  // This code works if `arr` and `bins` are monotonically increasing

  IndexArray indexes = IndexArray::Zero(arr.size());

  auto IsInsideBin = [arr, bins](Index item, Index index)
  {
    const double prc = 1.e-8;

    double a = arr(item);
    double bl = bins(index - 1);
    double br = bins(index);

    // bins[i-1] <= a < bins[i]
    return (a > bl || std::abs(a - bl) < std::abs(std::min(a, bl)) * prc) && a < br;
  };

  Index kstart = 1;

  for (Index i = 0; i < arr.size(); ++i) {
    for (Index k = kstart; k < bins.size(); ++k) {
      if (IsInsideBin(i, k)) {
        indexes(i) = k;
        kstart = k;
        break;
      }
    }
  }

  return indexes;
}

DoubleSparseMatrix MakeSparseDiagMatrix(const DoubleArray2D& diags, const IndexArray& offsets, Size rows, Size cols)
{
  auto GetNumElemsAndIndex = [rows, cols](Index offset, Index &i, Index &j)
  {
    if (offset < 0) {
      i = -offset;
      j = 0;
    }
    else {
      i = 0;
      j = offset;
    }

    return std::min(rows - i, cols - j);
  };

  DoubleSparseMatrix m(rows, cols);

  for (Index k = 0; k < offsets.size(); ++k) {
    auto offset = offsets(k);
    Index i, j;

    DoubleArray diag = diags.row(k);
    auto n = GetNumElemsAndIndex(offset, i, j);

    // When rows == cols or rows > cols, the function takes elements of the 
    // super-diagonal from the lower part of the corresponding diag array, and 
    // elements of the sub-diagonal from the upper part of the corresponding diag array.
    //
    // When rows < cols, the function does the opposite, taking elements of the 
    // super-diagonal from the upper part of the corresponding diag array, and 
    // elements of the sub-diagonal from the lower part of the corresponding diag array.

    if (offset < 0) {
      if (rows >= cols) {
        diag = diag.head(n);
      }
      else {
        diag = diag.tail(n);
      }
    }
    else {
      if (rows >= cols) {
        diag = diag.tail(n);
      }
      else {
        diag = diag.head(n);
      }
    }

    for (Index l = 0; l < n; ++l) {
      m.insert((int)(i+l), (int)(j+l)) = diag(l);
    }
  }

  return m;
}

csaps::DoubleArray SolveLinearSystem(const DoubleSparseMatrix &A, const DoubleArray &b)
{
  Eigen::SparseLU<DoubleSparseMatrix> solver;

  // Compute the ordering permutation vector from the structural pattern of A
  solver.analyzePattern(A);

  // Compute the numerical factorization
  solver.factorize(A);
  
  // Use the factors to solve the linear system
  DoubleArray x = solver.solve(b.matrix()).array();

  return x;
}

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

DoubleArray UnivariateCubicSmoothingSpline::operator()(const Size pcount, DoubleArray &xidata)
{
  if (pcount < 2) {
    throw std::exception("There must be at least 2 data points");
  }

  xidata.resize(pcount);
  xidata << DoubleArray::LinSpaced(pcount, m_xdata(0), m_xdata(m_xdata.size()-1));

  return Evaluate(xidata);
}

void UnivariateCubicSmoothingSpline::MakeSpline()
{
  const auto pcount = m_xdata.size();

  auto dx = Diff(m_xdata);
  auto dy = Diff(m_ydata);
  auto divdydx = dy / dx;

  double p = m_smooth;

  if (pcount > 2) {
    // Create diagonal sparse matrices
    const auto n = dx.size() - 1;

    DoubleArray2D diags(3, n);

    auto head_r = dx.head(n);
    auto tail_r = dx.tail(n);

    diags.row(0) = tail_r;
    diags.row(1) = 2 * (tail_r + head_r);
    diags.row(2) = head_r;
    
    IndexArray offsets(3); 
    
    offsets << -1, 0, 1;

    auto r = MakeSparseDiagMatrix(diags, offsets, pcount - 2, pcount - 2);

    auto odx = 1. / dx;

    auto head_qt = odx.head(n);
    auto tail_qt = odx.tail(n);

    diags.row(0) = head_qt;
    diags.row(1) = -(tail_qt + head_qt);
    diags.row(2) = tail_qt;

    offsets << 0, 1, 2;

    auto qt = MakeSparseDiagMatrix(diags, offsets, pcount - 2, pcount);

    DoubleArray ow = 1. / m_weights;
    DoubleArray osqw = 1. / m_weights.sqrt();
    
    offsets.resize(1);
    offsets << 0;

    auto w = MakeSparseDiagMatrix(ow.transpose(), offsets, pcount, pcount);
    auto qw = MakeSparseDiagMatrix(osqw.transpose(), offsets, pcount, pcount);
    
    DoubleSparseMatrix qtw = qt * qw;
    DoubleSparseMatrix qtwq = qtw * qtw.transpose();
    
    auto Trace = [](const DoubleSparseMatrix &m)
    {
      return m.diagonal().sum();
    };

    double p = m_smooth;

    if (p < 0) {
      p = 1. / (1. + Trace(r) / (6. * Trace(qtwq)));
    }
    
    DoubleSparseMatrix A = ((6. * (1. - p)) * qtwq) + (p * r);
    A.makeCompressed();

    auto b = Diff(divdydx);

    // Solve linear system Ab = u
    auto u = SolveLinearSystem(A, b);

  }
  else {
    p = 1.0;
    m_coeffs = DoubleArray2D(1, 2);
    m_coeffs(0, 0) = divdydx(0);
    m_coeffs(0, 1) = m_ydata(0);
  }

  m_smooth = p;
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

  for (Index i = 0; i < xi_size; ++i) {
    Index index = indexes(i);

    // Go to local coordinates
    xidata_loc(i) = xidata(i) - m_xdata(index);

    // Initial values
    yidata(i) = m_coeffs(index, 0);
  }

  DoubleArray coeffs(xi_size);

  for (Index i = 1; i < m_coeffs.cols(); ++i) {
    for (Index k = 0; k < xi_size; ++k) {
      coeffs(k) = m_coeffs(indexes(k), i);
    }

    yidata = xidata_loc * yidata + coeffs;
  }

  return yidata;
}

} // namespace csaps
