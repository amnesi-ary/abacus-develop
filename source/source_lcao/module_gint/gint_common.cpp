#include "gint_common.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_lcao/module_hcontainer/hcontainer_funcs.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/tool_quit.h"
#include <type_traits>

#ifdef __MPI
#include "source_base/module_external/blacs_connector.h"
#include <mpi.h>
#endif

namespace ModuleGint
{

template<typename Tout, typename Tin>
void cast_hcontainer_values(const HContainer<Tin>& src, HContainer<Tout>& dst)
{
    if (src.get_ijr_info() != dst.get_ijr_info()
        || src.is_gamma_only() != dst.is_gamma_only()
        || src.get_nnr() != dst.get_nnr())
    {
        ModuleBase::WARNING_QUIT("cast_hcontainer_values", "Source and destination HContainer shapes do not match.");
    }

    const Tin* src_values = src.get_wrapper();
    Tout* dst_values = dst.get_wrapper();
    if (src_values == nullptr || dst_values == nullptr)
    {
        ModuleBase::WARNING_QUIT("cast_hcontainer_values", "HContainer data buffer is not allocated.");
    }

    const size_t nnr = src.get_nnr();
    for (size_t i = 0; i < nnr; ++i)
    {
        dst_values[i] = static_cast<Tout>(src_values[i]);
    }
}

template<typename Tout, typename Tin>
HContainer<Tout> make_cast_hcontainer(const HContainer<Tin>& src)
{
    const auto ijr_info = src.get_ijr_info();

    if (src.get_paraV() != nullptr)
    {
        HContainer<Tout> dst(src.get_paraV(), nullptr, &ijr_info);
        if (src.is_gamma_only())
        {
            dst.fix_gamma();
        }
        cast_hcontainer_values(src, dst);
        return dst;
    }

    HContainer<Tout> dst(static_cast<int>(src.get_sparse_ap().size()));
    if (src.is_gamma_only())
    {
        dst.fix_gamma();
    }
    for (int iap = 0; iap < src.size_atom_pairs(); ++iap)
    {
        const auto& src_ap = src.get_atom_pair(iap);
        hamilt::AtomPair<Tout> dst_ap(src_ap.get_atom_i(), src_ap.get_atom_j());
        dst_ap.set_size(src_ap.get_col_size(), src_ap.get_row_size());
        for (int ir = 0; ir < src_ap.get_R_size(); ++ir)
        {
            const auto r_index = src_ap.get_R_index(ir);
            dst_ap.get_HR_values(r_index.x, r_index.y, r_index.z);
        }
        dst.insert_pair(dst_ap);
    }
    dst.allocate(nullptr, false);
    cast_hcontainer_values(src, dst);
    return dst;
}

template<typename T>
void compose_hr_gint(HContainer<T>& hr_gint)
{
    ModuleBase::TITLE("Gint", "compose_hr_gint");
    ModuleBase::timer::start("Gint", "compose_hr_gint");
    for (int iap = 0; iap < hr_gint.size_atom_pairs(); iap++)
    {
        auto& ap = hr_gint.get_atom_pair(iap);
        const int iat1 = ap.get_atom_i();
        const int iat2 = ap.get_atom_j();
        if (iat1 > iat2)
        {
            // fill lower triangle matrix with upper triangle matrix
            // the upper <IJR> is <iat2, iat1>
            const hamilt::AtomPair<T>* upper_ap = hr_gint.find_pair(iat2, iat1);
            const hamilt::AtomPair<T>* lower_ap = hr_gint.find_pair(iat1, iat2);
#ifdef __DEBUG
            assert(upper_ap != nullptr);
#endif
            for (int ir = 0; ir < ap.get_R_size(); ir++)
            {
                auto R_index = ap.get_R_index(ir);
                auto upper_mat = upper_ap->find_matrix(-R_index);
                auto lower_mat = lower_ap->find_matrix(R_index);
                for (int irow = 0; irow < upper_mat->get_row_size(); ++irow)
                {
                    for (int icol = 0; icol < upper_mat->get_col_size(); ++icol)
                    {
                        lower_mat->get_value(icol, irow) = upper_ap->get_value(irow, icol);
                    }
                }
            }
        }
    }
    ModuleBase::timer::end("Gint", "compose_hr_gint");
}

template <typename T>
void transfer_hr_gint_to_hR(const HContainer<T>& hr_gint, HContainer<T>& hR)
{
    ModuleBase::TITLE("Gint", "transfer_hr_gint_to_hR");
    ModuleBase::timer::start("Gint", "transfer_hr_gint_to_hR");
#ifdef __MPI
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size == 1)
    {
        hR.add(hr_gint);
    }
    else
    {
        hamilt::transferSerials2Parallels(hr_gint, &hR);
    }
#else
    hR.add(hr_gint);
#endif
    ModuleBase::timer::end("Gint", "transfer_hr_gint_to_hR");
}


namespace
{

struct SpinorComponentMap
{
    int real_part_idx;
    int imag_part_idx;
    int row_offset;
    int col_offset;
    std::complex<double> imag_factor;
};

const SpinorComponentMap SPINOR_COMPONENT_MAPS[4] = {
    {0, 3, 0, 0, std::complex<double>(1.0, 0.0)},
    {1, 2, 0, 1, std::complex<double>(0.0, 1.0)},
    {1, 2, 1, 0, std::complex<double>(0.0, -1.0)},
    {0, 3, 1, 1, std::complex<double>(-1.0, 0.0)}
};

struct DensitySpinBlockMap
{
    int row_offset;
    int col_offset;
};

const DensitySpinBlockMap DENSITY_SPIN_BLOCK_MAPS[4] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

std::vector<int> make_spinor_iat2iwt(const UnitCell& ucell)
{
    std::vector<int> iat2iwt(ucell.nat);
    for (int iat = 0; iat < ucell.nat; iat++)
    {
        iat2iwt[iat] = ucell.get_iat2iwt()[iat] / 2;
    }
    return iat2iwt;
}

hamilt::HContainer<std::complex<double>> make_spinor_gint_container(
    const UnitCell& ucell,
    const std::vector<int>& ijr_info)
{
    hamilt::HContainer<std::complex<double>> hRGint_tmp(ucell.nat);
    hRGint_tmp.insert_ijrs(&ijr_info, ucell);
    hRGint_tmp.allocate(nullptr, true);
    hRGint_tmp.set_zero();
    return hRGint_tmp;
}

void fill_lower_from_upper(
    const hamilt::AtomPair<std::complex<double>>& upper_ap,
    hamilt::AtomPair<std::complex<double>>* lower_ap,
    const ModuleBase::Vector3<int>& R_index)
{
    auto upper_mat = upper_ap.find_matrix(R_index);
    auto lower_mat = lower_ap->find_matrix(-R_index);
    for (int irow = 0; irow < upper_mat->get_row_size(); ++irow)
    {
        for (int icol = 0; icol < upper_mat->get_col_size(); ++icol)
        {
            lower_mat->get_value(icol, irow) = upper_mat->get_value(irow, icol);
        }
    }
}

void fill_spinor_component(
    const std::vector<hamilt::HContainer<double>>& hr_gint_tmp,
    const SpinorComponentMap& map,
    hamilt::HContainer<std::complex<double>>& hRGint_tmp)
{
    for (int iap = 0; iap < hRGint_tmp.size_atom_pairs(); iap++)
    {
        auto* upper_ap = &hRGint_tmp.get_atom_pair(iap);
        const int iat1 = upper_ap->get_atom_i();
        const int iat2 = upper_ap->get_atom_j();
        if (iat1 > iat2)
        {
            continue;
        }

        hamilt::AtomPair<std::complex<double>>* lower_ap = hRGint_tmp.find_pair(iat2, iat1);
        const hamilt::AtomPair<double>* ap_real = hr_gint_tmp[map.real_part_idx].find_pair(iat1, iat2);
        const hamilt::AtomPair<double>* ap_imag = hr_gint_tmp[map.imag_part_idx].find_pair(iat1, iat2);

        for (int ir = 0; ir < upper_ap->get_R_size(); ir++)
        {
            const auto R_index = upper_ap->get_R_index(ir);
            auto upper_mat = upper_ap->find_matrix(R_index);
            auto mat_real = ap_real->find_matrix(R_index);
            auto mat_imag = ap_imag->find_matrix(R_index);
            for (int irow = 0; irow < mat_real->get_row_size(); ++irow)
            {
                for (int icol = 0; icol < mat_real->get_col_size(); ++icol)
                {
                    upper_mat->get_value(irow, icol)
                        = mat_real->get_value(irow, icol) + map.imag_factor * mat_imag->get_value(irow, icol);
                }
            }

            // When component 0 or 3 is used, the real part does not need conjugation.
            // Components 1 and 2 are not Hermitian small matrices, so conjugation is also not applied.
            if (iat1 < iat2)
            {
                fill_lower_from_upper(*upper_ap, lower_ap, R_index);
            }
        }
    }
}

void merge_component_to_hR(
    const hamilt::HContainer<std::complex<double>>& hR_tmp,
    const SpinorComponentMap& map,
    hamilt::HContainer<std::complex<double>>& hR)
{
    for (int iap = 0; iap < hR.size_atom_pairs(); iap++)
    {
        auto* ap = &hR.get_atom_pair(iap);
        const int iat1 = ap->get_atom_i();
        const int iat2 = ap->get_atom_j();
        const auto* ap_nspin = hR_tmp.find_pair(iat1, iat2);
        for (int ir = 0; ir < ap->get_R_size(); ir++)
        {
            const auto R_index = ap->get_R_index(ir);
            auto upper_mat = ap->find_matrix(R_index);
            auto mat_nspin = ap_nspin->find_matrix(R_index);
            for (int irow = 0; irow < mat_nspin->get_row_size(); ++irow)
            {
                for (int icol = 0; icol < mat_nspin->get_col_size(); ++icol)
                {
                    upper_mat->get_value(2 * irow + map.row_offset, 2 * icol + map.col_offset)
                        = mat_nspin->get_value(irow, icol);
                }
            }
        }
    }
}

#ifdef __MPI
void init_spinor_parallel_orbitals(
    Parallel_Orbitals& pv,
    const UnitCell& ucell,
    const hamilt::HContainer<std::complex<double>>& hR,
    const std::vector<int>& iat2iwt)
{
    const int mg = hR.get_paraV()->get_global_row_size() / 2;
    const int ng = hR.get_paraV()->get_global_col_size() / 2;
    const int nb = hR.get_paraV()->get_block_size() / 2;
    const int blacs_ctxt = hR.get_paraV()->blacs_ctxt;

    pv.set(mg, ng, nb, blacs_ctxt);
    pv.set_atomic_trace(iat2iwt.data(), ucell.nat, mg);
}
#endif

template<typename TDM>
hamilt::HContainer<TDM> make_serial_density_spin_block_container(
    const UnitCell& ucell,
    const std::vector<int>& ijr_info)
{
    hamilt::HContainer<TDM> dm2d_tmp(ucell.nat);
    dm2d_tmp.insert_ijrs(&ijr_info, ucell);
    dm2d_tmp.allocate(nullptr, true);
    return dm2d_tmp;
}

#ifdef __MPI
template<typename TDM>
void init_density_parallel_orbitals(
    Parallel_Orbitals& pv,
    const UnitCell& ucell,
    const hamilt::HContainer<TDM>& dm_spinor,
    const std::vector<int>& iat2iwt)
{
    const int mg = dm_spinor.get_paraV()->get_global_row_size() / 2;
    const int ng = dm_spinor.get_paraV()->get_global_col_size() / 2;
    const int nb = dm_spinor.get_paraV()->get_block_size() / 2;
    const int blacs_ctxt = dm_spinor.get_paraV()->blacs_ctxt;

    pv.set(mg, ng, nb, blacs_ctxt);
    pv.set_atomic_trace(iat2iwt.data(), ucell.nat, mg);
}
#endif

template<typename TDM>
inline void copy_density_spin_block(
    const TDM* matrix_in,
    TDM* matrix_out,
    const int row_size,
    const int col_size,
    const DensitySpinBlockMap& map)
{
    const int block_row_size = row_size / 2;
    const int block_col_size = col_size / 2;
    for (int irow = 0; irow < block_row_size; irow++)
    {
        for (int icol = 0; icol < block_col_size; icol++)
        {
            const int index_out = irow * block_col_size + icol;
            const int index_in = (irow * 2 + map.row_offset) * col_size + icol * 2 + map.col_offset;
            matrix_out[index_out] = matrix_in[index_in];
        }
    }
}

template<typename TDM>
void extract_density_spin_block(
    const hamilt::HContainer<TDM>& dm_spinor,
    const DensitySpinBlockMap& map,
    hamilt::HContainer<TDM>& dm2d_tmp)
{
    for (int iap = 0; iap < dm_spinor.size_atom_pairs(); ++iap)
    {
        const auto& ap = dm_spinor.get_atom_pair(iap);
        const int iat1 = ap.get_atom_i();
        const int iat2 = ap.get_atom_j();
        const int row_size = ap.get_row_size();
        const int col_size = ap.get_col_size();
        for (int ir = 0; ir < ap.get_R_size(); ++ir)
        {
            const ModuleBase::Vector3<int> r_index = ap.get_R_index(ir);
            TDM* matrix_out = dm2d_tmp.find_matrix(iat1, iat2, r_index)->get_pointer();
            TDM* matrix_in = ap.get_pointer(ir);
            copy_density_spin_block(matrix_in, matrix_out, row_size, col_size, map);
        }
    }
}

} // namespace

void merge_hr_part_to_hR(const std::vector<hamilt::HContainer<double>>& hr_gint_tmp,
                         hamilt::HContainer<std::complex<double>>* hR,
                         const GintInfo& gint_info)
{
    ModuleBase::TITLE("Gint_k", "transfer_pvpR");
    ModuleBase::timer::start("Gint_k", "transfer_pvpR");

    const UnitCell* ucell_in = gint_info.get_ucell();

#ifdef __MPI
    const auto iat2iwt = make_spinor_iat2iwt(*ucell_in);
    Parallel_Orbitals pv;
    init_spinor_parallel_orbitals(pv, *ucell_in, *hR, iat2iwt);
    auto ijr_info = hR->get_ijr_info();
    hamilt::HContainer<std::complex<double>> hR_tmp(&pv, nullptr, &ijr_info);
#endif

    for (int is = 0; is < 4; is++)
    {
        if (!PARAM.globalv.domag && (is == 1 || is == 2))
        {
            continue;
        }

        const auto& map = SPINOR_COMPONENT_MAPS[is];
        auto hRGint_tmp = make_spinor_gint_container(*ucell_in, gint_info.get_ijr_info());

        fill_spinor_component(hr_gint_tmp, map, hRGint_tmp);

#ifdef __MPI
        hR_tmp.set_zero();
        hamilt::transferSerials2Parallels(hRGint_tmp, &hR_tmp);
        merge_component_to_hR(hR_tmp, map, *hR);
#else
        merge_component_to_hR(hRGint_tmp, map, *hR);
#endif
    }

    ModuleBase::timer::end("Gint_k", "transfer_pvpR");
}



// C++11-compatible helpers for transfer_dm_2d_to_gint:
// SFINAE (enable_if) to select same-type vs cross-type code paths at compile time.

// Same-type path (TGint == TDM): transfer directly
template<typename TGint, typename TDM>
typename std::enable_if<std::is_same<TGint, TDM>::value>::type
gather_dm(const HContainer<TDM>& dm_src, HContainer<TGint>& dm_dst,
          const GintInfo& /*gint_info*/)
{
#ifdef __MPI
    hamilt::transferParallels2Serials(dm_src, &dm_dst);
#else
    dm_dst.set_zero();
    dm_dst.add(dm_src);
#endif
}

// Cross-type path (TGint != TDM): gather into a temp of TDM, then cast
template<typename TGint, typename TDM>
typename std::enable_if<!std::is_same<TGint, TDM>::value>::type
gather_dm(const HContainer<TDM>& dm_src, HContainer<TGint>& dm_dst,
          const GintInfo& gint_info)
{
    HContainer<TDM> dm_tmp = gint_info.get_hr<TDM>();
#ifdef __MPI
    hamilt::transferParallels2Serials(dm_src, &dm_tmp);
#else
    dm_tmp.set_zero();
    dm_tmp.add(dm_src);
#endif
    cast_hcontainer_values(dm_tmp, dm_dst);
}

template<typename TGint, typename TDM>
void transfer_density_components(
    const GintInfo& gint_info,
    const std::vector<HContainer<TDM>*>& dm,
    std::vector<HContainer<TGint>>& dm_gint)
{
    // dm_gint.size() usually equals to PARAM.inp.nspin,
    // but there is exception within source_lcao/module_lr.
    for (int is = 0; is < dm_gint.size(); is++)
    {
        gather_dm(*dm[is], dm_gint[is], gint_info);
    }
}

template<typename TGint, typename TDM>
void transfer_density_spinor_components(
    const GintInfo& gint_info,
    const std::vector<HContainer<TDM>*>& dm,
    std::vector<HContainer<TGint>>& dm_gint)
{
    const UnitCell* ucell = gint_info.get_ucell();
    auto ijr_info = dm[0]->get_ijr_info();

#ifdef __MPI
    const auto iat2iwt = make_spinor_iat2iwt(*ucell);
    Parallel_Orbitals pv{};
    init_density_parallel_orbitals(pv, *ucell, *dm[0], iat2iwt);
    HContainer<TDM> dm2d_tmp(&pv, nullptr, &ijr_info);
#else
    auto dm2d_tmp = make_serial_density_spin_block_container<TDM>(*ucell, ijr_info);
#endif

    for (int is = 0; is < 4; is++)
    {
        extract_density_spin_block(*dm[0], DENSITY_SPIN_BLOCK_MAPS[is], dm2d_tmp);
        gather_dm(dm2d_tmp, dm_gint[is], gint_info);
    }
}

// gint_info should not have been a parameter, but it was added to initialize dm_gint_full
// In the future, we might try to remove the gint_info parameter
template<typename TGint, typename TDM>
void transfer_dm_2d_to_gint(
    const GintInfo& gint_info,
    const std::vector<HContainer<TDM>*>& dm,
    std::vector<HContainer<TGint>>& dm_gint)
{
    ModuleBase::TITLE("Gint", "transfer_dm_2d_to_gint");
    ModuleBase::timer::start("Gint", "transfer_dm_2d_to_gint");

    if (PARAM.inp.nspin != 4)
    {
        transfer_density_components(gint_info, dm, dm_gint);
    } else  // NSPIN=4 case
    {
        transfer_density_spinor_components(gint_info, dm, dm_gint);
    }
    ModuleBase::timer::end("Gint", "transfer_dm_2d_to_gint");
}

int globalIndex(int localindex, int nblk, int nprocs, int myproc)
{
    const int iblock = localindex / nblk;
    const int gIndex = (iblock * nprocs + myproc) * nblk + localindex % nblk;
    return gIndex;
}

int localIndex(int globalindex, int nblk, int nprocs, int& myproc)
{
    myproc = int((globalindex % (nblk * nprocs)) / nblk);
    return int(globalindex / (nblk * nprocs)) * nblk + globalindex % nblk;
}

template <typename T>
void wfc_2d_to_gint(const T* wfc_2d,
                    int nbands,  // needed if MPI is disabled
                    int nlocal,  // needed if MPI is disabled
                    const Parallel_Orbitals& pv,
                    T* wfc_gint,
                    const GintInfo& gint_info)
{
    ModuleBase::TITLE("Gint", "wfc_2d_to_gint");
    ModuleBase::timer::start("Gint", "wfc_2d_to_gint");

#ifdef __MPI
    // dimension related
    nlocal = pv.desc_wfc[2];
    nbands = pv.desc_wfc[3];

    const std::vector<int>& trace_lo = gint_info.get_trace_lo();

    // MPI and memory related
    const int mem_stride = 1;
    int mpi_info = 0;

    // get the rank of the current process
    int rank = 0;
    MPI_Comm_rank(pv.comm(), &rank);

    // calculate the maximum number of nlocal over all processes in pv.comm() range
    long buf_size = 0;
    mpi_info = MPI_Reduce(&pv.nloc_wfc, &buf_size, 1, MPI_LONG, MPI_MAX, 0, pv.comm());
    mpi_info = MPI_Bcast(&buf_size, 1, MPI_LONG, 0, pv.comm()); // get and then broadcast
    std::vector<T> wfc_block(buf_size);

    // this quantity seems to have the value returned by function numroc_ in ScaLAPACK?
    int naroc[2];

    // for BLACS broadcast
    char scope = 'A';
    char top = ' ';

    // loop over all processors
    for (int iprow = 0; iprow < pv.dim0; ++iprow)
    {
        for (int ipcol = 0; ipcol < pv.dim1; ++ipcol)
        {
            if (iprow == pv.coord[0] && ipcol == pv.coord[1])
            {
                BlasConnector::copy(pv.nloc_wfc, wfc_2d, mem_stride, wfc_block.data(), mem_stride);
                naroc[0] = pv.nrow;
                naroc[1] = pv.ncol_bands;
                Cxgebs2d(pv.blacs_ctxt, &scope, &top, 2, 1, naroc, 2);
                Cxgebs2d(pv.blacs_ctxt, &scope, &top, buf_size, 1, wfc_block.data(), buf_size);
            }
            else
            {
                Cxgebr2d(pv.blacs_ctxt, &scope, &top, 2, 1, naroc, 2, iprow, ipcol);
                Cxgebr2d(pv.blacs_ctxt, &scope, &top, buf_size, 1, wfc_block.data(), buf_size, iprow, ipcol);
            }

            // then use it to set the wfc_grid.
            const int nb = pv.nb;
            const int dim0 = pv.dim0;
            const int dim1 = pv.dim1;
            for (int j = 0; j < naroc[1]; ++j)
            {
                int igcol = globalIndex(j, nb, dim1, ipcol);
                if (igcol >= PARAM.inp.nbands)
                {
                    continue;
                }
                for (int i = 0; i < naroc[0]; ++i)
                {
                    int igrow = globalIndex(i, nb, dim0, iprow);
                    int mu_local = trace_lo[igrow];
                    if (wfc_gint && mu_local >= 0)
                    {
                        wfc_gint[igcol * nlocal + mu_local] = wfc_block[j * naroc[0] + i];
                    }
                }
            }
            // this operation will let all processors have the same wfc_grid
        }
    }
#else
    for (int i = 0; i < nbands; ++i)
    {
        for (int j = 0; j < nlocal; ++j)
        {
            wfc_gint[i * nlocal + j] = wfc_2d[i * nlocal + j];
        }
    }
#endif
    ModuleBase::timer::end("Gint", "wfc_2d_to_gint");
}

template void compose_hr_gint(HContainer<double>& hr_gint);
template void compose_hr_gint(HContainer<float>& hr_gint);
template void transfer_hr_gint_to_hR(
    const HContainer<double>& hr_gint,
    HContainer<double>& hR);
template void transfer_hr_gint_to_hR(
    const HContainer<std::complex<double>>& hr_gint,
    HContainer<std::complex<double>>& hR);
template void cast_hcontainer_values(
    const HContainer<double>& src,
    HContainer<float>& dst);
template void cast_hcontainer_values(
    const HContainer<float>& src,
    HContainer<double>& dst);
template HContainer<float> make_cast_hcontainer(const HContainer<double>& src);
template HContainer<double> make_cast_hcontainer(const HContainer<float>& src);
template void transfer_dm_2d_to_gint(
    const GintInfo& gint_info,
    const std::vector<HContainer<double>*>& dm,
    std::vector<HContainer<double>>& dm_gint);
template void transfer_dm_2d_to_gint(
    const GintInfo& gint_info,
    const std::vector<HContainer<double>*>& dm,
    std::vector<HContainer<float>>& dm_gint);
template void transfer_dm_2d_to_gint(
    const GintInfo& gint_info,
    const std::vector<HContainer<std::complex<double>>*>& dm,
    std::vector<HContainer<std::complex<double>>>& dm_gint);
template void wfc_2d_to_gint(
    const double* wfc_2d,
    int nbands,
    int nlocal,
    const Parallel_Orbitals& pv,
    double* wfc_grid,
    const GintInfo& gint_info);
template void wfc_2d_to_gint(
    const std::complex<double>* wfc_2d,
    int nbands,
    int nlocal,
    const Parallel_Orbitals& pv,
    std::complex<double>* wfc_grid,
    const GintInfo& gint_info);
}
