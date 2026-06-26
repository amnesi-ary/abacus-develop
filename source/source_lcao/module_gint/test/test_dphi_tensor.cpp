#include "gtest/gtest.h"

#include "../dphi_tensor.h"

TEST(DphiTensorTest, StoresDisplacementsInContiguousOrder)
{
    using ModuleGint::detail::DphiTensor;

    DphiTensor tensor(2);

    EXPECT_EQ(tensor.nw(), 2);
    EXPECT_EQ(tensor.size(), 2 * 6 * 3);

    for (int iw = 0; iw < tensor.nw(); ++iw)
    {
        for (int displacement = 0; displacement < DphiTensor::kDisplacements; ++displacement)
        {
            for (int direction = 0; direction < DphiTensor::kDirections; ++direction)
            {
                tensor(iw, displacement, direction) = 100.0 * iw + 10.0 * displacement + direction;
            }
        }
    }

    for (int iw = 0; iw < tensor.nw(); ++iw)
    {
        for (int displacement = 0; displacement < DphiTensor::kDisplacements; ++displacement)
        {
            for (int direction = 0; direction < DphiTensor::kDirections; ++direction)
            {
                const int flat_index = (iw * DphiTensor::kDisplacements + displacement) * DphiTensor::kDirections
                    + direction;
                const double expected = 100.0 * iw + 10.0 * displacement + direction;
                EXPECT_DOUBLE_EQ(tensor(iw, displacement, direction), expected);
                EXPECT_DOUBLE_EQ(tensor.data()[flat_index], expected);
            }
        }
    }
}
