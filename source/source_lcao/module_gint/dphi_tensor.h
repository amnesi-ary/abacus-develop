#pragma once

#include <cstddef>
#include <vector>

namespace ModuleGint
{
namespace detail
{

class DphiTensor
{
  public:
    static constexpr int kDisplacements = 6;
    static constexpr int kDirections = 3;

    explicit DphiTensor(const int nw)
        : nw_(nw), data_(static_cast<std::size_t>(nw) * kDisplacements * kDirections)
    {}

    double& operator()(const int iw, const int displacement, const int direction)
    {
        return data_[index(iw, displacement, direction)];
    }

    const double& operator()(const int iw, const int displacement, const int direction) const
    {
        return data_[index(iw, displacement, direction)];
    }

    int nw() const
    {
        return nw_;
    }

    std::size_t size() const
    {
        return data_.size();
    }

    double* data()
    {
        return data_.data();
    }

    const double* data() const
    {
        return data_.data();
    }

  private:
    static std::size_t index(const int iw, const int displacement, const int direction)
    {
        return static_cast<std::size_t>((iw * kDisplacements + displacement) * kDirections + direction);
    }

    int nw_ = 0;
    std::vector<double> data_;
};

} // namespace detail
} // namespace ModuleGint
