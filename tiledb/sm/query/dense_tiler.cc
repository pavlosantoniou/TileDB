/**
 * @file   dense_tiler.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2021 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * This file implements class DenseTiler.
 */

#include "tiledb/sm/query/dense_tiler.h"
#include "tiledb/sm/array/array.h"
#include "tiledb/sm/array_schema/dimension.h"
#include "tiledb/sm/array_schema/domain.h"
#include "tiledb/sm/misc/constants.h"
#include "tiledb/sm/tile/tile.h"

using namespace tiledb::common;

namespace tiledb {
namespace sm {

/* ****************************** */
/*   CONSTRUCTORS & DESTRUCTORS   */
/* ****************************** */

template <class T>
DenseTiler<T>::DenseTiler(
    const std::unordered_map<std::string, QueryBuffer>& buffers,
    const Subarray* subarray)
    : array_schema_(subarray->array()->array_schema())
    , buffers_(buffers)
    , subarray_(subarray) {
  // Assertions
  assert(subarray != nullptr);
  for (const auto& buff : buffers)
    assert(array_schema_->is_attr(buff.first));

  // Initializations
  calculate_tile_num();
  calculate_subarray_tile_coord_offsets();
  calculate_first_sub_tile_coords();
  calculate_tile_and_subarray_strides();
}

template <class T>
DenseTiler<T>::~DenseTiler() {
}

/* ****************************** */
/*               API              */
/* ****************************** */

template <class T>
const typename DenseTiler<T>::CopyPlan DenseTiler<T>::copy_plan(
    uint64_t id) const {
  // For easy reference
  CopyPlan ret;
  auto dim_num = (int32_t)array_schema_->dim_num();
  auto domain = array_schema_->domain();
  auto subarray = subarray_->ndrange(0);  // Guaranteed to be unary
  std::vector<std::array<T, 2>> sub(dim_num);
  for (int32_t d = 0; d < dim_num; ++d)
    sub[d] = {*(const T*)subarray[d].start(), *(const T*)subarray[d].end()};
  auto tile_layout = array_schema_->cell_order();
  auto sub_layout = subarray_->layout();

  // Copy tile and subarray strides
  ret.tile_strides_el_ = tile_strides_el_;
  ret.sub_strides_el_ = sub_strides_el_;

  // Focus on the input tile
  auto tile_sub = this->tile_subarray(id);
  auto sub_in_tile = utils::geometry::intersection<T>(sub, tile_sub);

  // Compute the starting element to copy from in the subarray, and
  // to copy to in the tile
  ret.sub_start_el_ = 0;
  ret.tile_start_el_ = 0;
  for (int32_t d = 0; d < dim_num; ++d) {
    ret.sub_start_el_ += (sub_in_tile[d][0] - sub[d][0]) * sub_strides_el_[d];
    ret.tile_start_el_ +=
        (sub_in_tile[d][0] - tile_sub[d][0]) * tile_strides_el_[d];
  }

  // Calculate the copy elements per iteration, as well as the
  // dimension ranges to focus on
  if (dim_num == 1) {  // Special case, copy the entire subarray 1D range
    ret.dim_ranges_.push_back({0, 0});
    ret.copy_el_ = sub_in_tile[0][1] - sub_in_tile[0][0] + 1;
  } else if (sub_layout != tile_layout) {
    ret.copy_el_ = 1;
    for (int32_t d = 0; d < dim_num; ++d) {
      ret.dim_ranges_.push_back(
          {(uint64_t)0, uint64_t(sub_in_tile[d][1] - sub_in_tile[d][0])});
    }
  } else {  // dim_num > 1 && same layout of tile and subarray cells
    if (tile_layout == Layout::ROW_MAJOR) {
      ret.copy_el_ =
          sub_in_tile[dim_num - 1][1] - sub_in_tile[dim_num - 1][0] + 1;
      int32_t last_d = dim_num - 2;
      for (; last_d >= 0; --last_d) {
        auto tile_extent = *(const T*)domain->tile_extent(last_d + 1).data();
        if (sub_in_tile[last_d + 1][1] - sub_in_tile[last_d + 1][0] + 1 ==
                tile_extent &&
            sub_in_tile[last_d + 1][0] == sub[last_d + 1][0] &&
            sub_in_tile[last_d + 1][1] == sub[last_d + 1][1])
          ret.copy_el_ *= sub_in_tile[last_d][1] - sub_in_tile[last_d][0] + 1;
        else
          break;
      }
      if (last_d < 0) {
        ret.dim_ranges_.push_back({0, 0});
      } else {
        for (int32_t d = 0; d <= last_d; ++d)
          ret.dim_ranges_.push_back(
              {0, (uint64_t)(sub_in_tile[d][1] - sub_in_tile[d][0])});
      }
    } else {  // COL_MAJOR
      ret.copy_el_ = sub_in_tile[0][1] - sub_in_tile[0][0] + 1;
      int32_t last_d = 1;
      for (; last_d < dim_num; ++last_d) {
        auto tile_extent = *(const T*)domain->tile_extent(last_d - 1).data();
        if (sub_in_tile[last_d - 1][1] - sub_in_tile[last_d - 1][0] + 1 ==
                tile_extent &&
            sub_in_tile[last_d - 1][0] == sub[last_d - 1][0] &&
            sub_in_tile[last_d - 1][1] == sub[last_d - 1][1])
          ret.copy_el_ *= sub_in_tile[last_d][1] - sub_in_tile[last_d][0] + 1;
        else
          break;
      }
      if (last_d == dim_num) {
        ret.dim_ranges_.push_back({0, 0});
      } else {
        for (int32_t d = last_d; d < dim_num; ++d)
          ret.dim_ranges_.push_back(
              {0, (uint64_t)(sub_in_tile[d][1] - sub_in_tile[d][0])});
      }
    }
  }

  return ret;
}

template <class T>
Status DenseTiler<T>::get_tile(
    uint64_t id, const std::string& name, Tile* tile) const {
  // Checks
  if (id >= tile_num_)
    return LOG_STATUS(
        Status::DenseTilerError("Cannot get tile; Invalid tile id"));
  if (!array_schema_->is_attr(name))
    return LOG_STATUS(Status::DenseTilerError(
        std::string("Cannot get tile; '") + name + "' is not an attribute"));
  if (array_schema_->var_size(name))
    return LOG_STATUS(Status::DenseTilerError(
        std::string("Cannot get tile; '") + name +
        "' is not a fixed-sized attribute"));

  // Init and fill entire tile with the fill values
  RETURN_NOT_OK(init_tile(name, tile));
  RETURN_NOT_OK(fill_tile(name, tile));

  // Calculate copy plan
  const CopyPlan copy_plan = this->copy_plan(id);

  // For easy reference
  auto cell_size = array_schema_->cell_size(name);
  auto sub_offset = copy_plan.sub_start_el_ * cell_size;
  auto tile_offset = copy_plan.tile_start_el_ * cell_size;
  auto copy_nbytes = copy_plan.copy_el_ * cell_size;
  auto sub_strides_nbytes = copy_plan.sub_strides_el_;
  for (auto& bsn : sub_strides_nbytes)
    bsn *= cell_size;
  auto tile_strides_nbytes = copy_plan.tile_strides_el_;
  for (auto& tsn : tile_strides_nbytes)
    tsn *= cell_size;
  auto buff = (uint8_t*)buffers_.get().find(name)->second.buffer_;
  const auto& dim_ranges = copy_plan.dim_ranges_;
  auto dim_num = (int64_t)dim_ranges.size();
  assert(dim_num > 0);

  // Auxiliary information needed in the copy loop
  std::vector<uint64_t> tile_offsets(dim_num);
  for (int64_t i = 0; i < dim_num; ++i)
    tile_offsets[i] = tile_offset;
  std::vector<uint64_t> sub_offsets(dim_num);
  for (int64_t i = 0; i < dim_num; ++i)
    sub_offsets[i] = sub_offset;
  std::vector<uint64_t> cell_coords(dim_num);
  for (int64_t i = 0; i < dim_num; ++i)
    cell_coords[i] = dim_ranges[i][0];

  // Perform the tile copy (always in row-major order)
  auto d = dim_num - 1;
  while (true) {
    // Copy a slab
    RETURN_NOT_OK(
        tile->write(&buff[sub_offsets[d]], tile_offsets[d], copy_nbytes));

    // Advance cell coordinates, tile and buffer offsets
    auto last_dim_changed = d;
    for (; last_dim_changed >= 0; --last_dim_changed) {
      ++cell_coords[last_dim_changed];
      if (cell_coords[last_dim_changed] > dim_ranges[last_dim_changed][1])
        cell_coords[last_dim_changed] = dim_ranges[last_dim_changed][0];
      else
        break;
    }

    // Check if copy loop is done
    if (last_dim_changed < 0)
      break;

    // Update the offsets
    tile_offsets[last_dim_changed] += tile_strides_nbytes[last_dim_changed];
    sub_offsets[last_dim_changed] += sub_strides_nbytes[last_dim_changed];
    for (auto i = last_dim_changed + 1; i < dim_num; ++i) {
      tile_offsets[i] = tile_offsets[i - 1];
      sub_offsets[i] = sub_offsets[i - 1];
    }
  }

  // Reset the tile offset to the beginning of the tile
  tile->reset_offset();

  return Status::Ok();
}

template <class T>
Status DenseTiler<T>::get_tile(
    uint64_t id,
    const std::string& name,
    Tile* tile_off,
    Tile* tile_val) const {
  (void)id;
  (void)name;
  (void)tile_off;
  (void)tile_val;

  return Status::Ok();
}

template <class T>
uint64_t DenseTiler<T>::tile_num() const {
  return tile_num_;
}

template <class T>
const std::vector<uint64_t>& DenseTiler<T>::tile_strides_el() const {
  return tile_strides_el_;
}

template <class T>
const std::vector<uint64_t>& DenseTiler<T>::sub_strides_el() const {
  return sub_strides_el_;
}

template <class T>
const std::vector<uint64_t>& DenseTiler<T>::sub_tile_coord_offsets() const {
  return sub_tile_coord_offsets_;
}

template <class T>
const std::vector<uint64_t>& DenseTiler<T>::first_sub_tile_coords() const {
  return first_sub_tile_coords_;
}

/* ****************************** */
/*          PRIVATE METHODS       */
/* ****************************** */

template <class T>
void DenseTiler<T>::calculate_first_sub_tile_coords() {
  // For easy reference
  auto dim_num = array_schema_->dim_num();
  auto domain = array_schema_->domain();
  auto subarray = subarray_->ndrange(0);

  // Calculate the coordinates of the first tile in the entire
  // domain that intersects the subarray (essentially its upper
  // left cell)
  first_sub_tile_coords_.resize(dim_num);
  for (unsigned d = 0; d < dim_num; ++d) {
    T dom_start = *(const T*)domain->dimension(d)->domain().start();
    T sub_start = *(const T*)subarray[d].start();
    T tile_extent = *(const T*)domain->tile_extent(d).data();
    first_sub_tile_coords_[d] = (sub_start - dom_start) / tile_extent;
  }
}

template <class T>
void DenseTiler<T>::calculate_subarray_tile_coord_offsets() {
  // For easy reference
  auto dim_num = (int32_t)array_schema_->dim_num();
  auto domain = array_schema_->domain();
  auto subarray = subarray_->ndrange(0);
  auto layout = array_schema_->tile_order();

  // Compute offsets
  sub_tile_coord_offsets_.reserve(dim_num);
  if (layout == Layout::ROW_MAJOR) {
    sub_tile_coord_offsets_.push_back(1);
    for (auto d = dim_num - 2; d >= 0; --d) {
      auto tile_num = domain->dimension(d)->tile_num(subarray[d]);
      sub_tile_coord_offsets_.push_back(
          sub_tile_coord_offsets_.back() * tile_num);
    }
    std::reverse(
        sub_tile_coord_offsets_.begin(), sub_tile_coord_offsets_.end());
  } else {  // COL_MAJOR
    sub_tile_coord_offsets_.push_back(1);
    if (dim_num > 1) {
      for (int32_t d = 1; d < dim_num; ++d) {
        auto tile_num = domain->dimension(d)->tile_num(subarray[d]);
        sub_tile_coord_offsets_.push_back(
            sub_tile_coord_offsets_.back() * tile_num);
      }
    }
  }
}

template <class T>
void DenseTiler<T>::calculate_tile_and_subarray_strides() {
  // For easy reference
  auto sub_layout = subarray_->layout();
  assert(sub_layout == Layout::ROW_MAJOR || sub_layout == Layout::COL_MAJOR);
  auto tile_layout = array_schema_->cell_order();
  auto dim_num = (int32_t)array_schema_->dim_num();
  auto domain = array_schema_->domain();
  auto subarray = subarray_->ndrange(0);

  // Compute tile strides
  tile_strides_el_.resize(dim_num);
  if (tile_layout == Layout::ROW_MAJOR) {
    tile_strides_el_[dim_num - 1] = 1;
    if (dim_num > 1) {
      for (auto d = dim_num - 2; d >= 0; --d) {
        auto tile_extent = (const T*)(&domain->tile_extent(d + 1)[0]);
        assert(tile_extent != nullptr);
        tile_strides_el_[d] = tile_strides_el_[d + 1] * *tile_extent;
      }
    }
  } else {  // COL_MAJOR
    tile_strides_el_[0] = 1;
    if (dim_num > 1) {
      for (auto d = 1; d < dim_num; ++d) {
        auto tile_extent = (const T*)(&domain->tile_extent(d - 1)[0]);
        assert(tile_extent != nullptr);
        tile_strides_el_[d] = tile_strides_el_[d - 1] * *tile_extent;
      }
    }
  }

  // Compute subarray slides
  sub_strides_el_.resize(dim_num);
  if (sub_layout == Layout::ROW_MAJOR) {
    sub_strides_el_[dim_num - 1] = 1;
    if (dim_num > 1) {
      for (auto d = dim_num - 2; d >= 0; --d) {
        auto sub_range_start = *(const T*)subarray[d + 1].start();
        auto sub_range_end = *(const T*)subarray[d + 1].end();
        auto sub_extent = sub_range_end - sub_range_start + 1;
        sub_strides_el_[d] = sub_strides_el_[d + 1] * sub_extent;
      }
    }
  } else {  // COL_MAJOR
    sub_strides_el_[0] = 1;
    if (dim_num > 1) {
      for (auto d = 1; d < dim_num; ++d) {
        auto sub_range_start = *(const T*)subarray[d - 1].start();
        auto sub_range_end = *(const T*)subarray[d - 1].end();
        auto sub_extent = sub_range_end - sub_range_start + 1;
        sub_strides_el_[d] = sub_strides_el_[d - 1] * sub_extent;
      }
    }
  }
}

template <class T>
void DenseTiler<T>::calculate_tile_num() {
  tile_num_ = array_schema_->domain()->tile_num(subarray_->ndrange(0));
}

template <class T>
Status DenseTiler<T>::fill_tile(const std::string& name, Tile* tile) const {
  // For easy reference
  auto attr = array_schema_->attribute(name);
  const void* fill_value;
  uint64_t fill_size;
  RETURN_NOT_OK(attr->get_fill_value(&fill_value, &fill_size));
  auto cell_num = array_schema_->domain()->cell_num_per_tile();

  // Prepare batch information
  uint64_t batch_cell_num = 1000000;  // This can change
  uint64_t last_batch_cell_num = cell_num % batch_cell_num;
  uint64_t batch_num = cell_num / 1000000 + (last_batch_cell_num > 0);
  uint64_t batch_size = batch_cell_num * fill_size;
  std::vector<uint8_t> batch;
  batch.resize(batch_size);
  uint64_t offset = 0;
  for (uint64_t i = 0; i < batch_cell_num; ++i) {
    memcpy(&batch[offset], fill_value, fill_size);
    offset += fill_size;
  }

  // Fill the tile, one batch at a time (instead of a cell at a time)
  if (batch_num > 1) {
    for (uint64_t b = 0; b < batch_num - 1; ++b)
      tile->write(batch.data(), batch_size);
  }

  // Fill the last batch
  if (last_batch_cell_num > 0) {
    for (uint64_t i = 0; i < last_batch_cell_num; ++i)
      tile->write(fill_value, fill_size);
  }

  // Sanity checks
  assert(cell_num * fill_size == tile->size());
  assert(tile->size() == tile->offset());

  // Reset offset so that writing
  tile->reset_offset();

  return Status::Ok();
}

template <class T>
Status DenseTiler<T>::init_tile(const std::string& name, Tile* tile) const {
  // For easy reference
  auto cell_size = array_schema_->cell_size(name);
  auto type = array_schema_->type(name);
  auto domain = array_schema_->domain();
  auto cell_num_per_tile = domain->cell_num_per_tile();
  auto tile_size = cell_num_per_tile * cell_size;

  // Initialize
  RETURN_NOT_OK(tile->init_unfiltered(
      constants::format_version, type, tile_size, cell_size, 0));

  return Status::Ok();
}

template <class T>
std::vector<uint64_t> DenseTiler<T>::tile_coords_in_sub(uint64_t id) const {
  // For easy reference
  auto dim_num = (int32_t)array_schema_->dim_num();
  auto layout = array_schema_->tile_order();
  std::vector<uint64_t> ret;
  auto tmp_idx = id;

  if (layout == Layout::ROW_MAJOR) {
    for (int32_t d = 0; d < dim_num; ++d) {
      ret.push_back(tmp_idx / sub_tile_coord_offsets_[d]);
      tmp_idx %= sub_tile_coord_offsets_[d];
    }
  } else {  // COL_MAJOR
    for (auto d = dim_num - 1; d >= 0; --d) {
      ret.push_back(tmp_idx / sub_tile_coord_offsets_[d]);
      tmp_idx %= sub_tile_coord_offsets_[d];
    }
    std::reverse(ret.begin(), ret.end());
  }

  return ret;
}

template <class T>
std::vector<std::array<T, 2>> DenseTiler<T>::tile_subarray(uint64_t id) const {
  // For easy reference
  auto dim_num = array_schema_->dim_num();
  auto domain = array_schema_->domain();

  // Get tile coordinates in the subarray tile domain
  auto tile_coords_in_sub = this->tile_coords_in_sub(id);

  // Get the tile coordinates in the array tile domain
  std::vector<uint64_t> tile_coords_in_dom(dim_num);
  for (unsigned d = 0; d < dim_num; ++d)
    tile_coords_in_dom[d] = tile_coords_in_sub[d] + first_sub_tile_coords_[d];

  // Calculate tile subarray based on the tile coordinates in the domain
  std::vector<std::array<T, 2>> ret(dim_num);
  for (unsigned d = 0; d < dim_num; ++d) {
    auto dom_start = *(const T*)domain->dimension(d)->domain().start();
    auto tile_extent = *(const T*)domain->tile_extent(d).data();
    ret[d][0] = tile_coords_in_dom[d] * tile_extent + dom_start;
    ret[d][1] = ret[d][0] + tile_extent - 1;
  }

  return ret;
}

// Explicit template instantiations
template class DenseTiler<int8_t>;
template class DenseTiler<uint8_t>;
template class DenseTiler<int16_t>;
template class DenseTiler<uint16_t>;
template class DenseTiler<int32_t>;
template class DenseTiler<uint32_t>;
template class DenseTiler<int64_t>;
template class DenseTiler<uint64_t>;

}  // namespace sm
}  // namespace tiledb

// TODO: nullables?
// TODO: var-sized
// TODO: does it make sense to parllelize the tile copy?
// TODO: clean dead code
// TODO: unify term "offsets" vs. "strides"