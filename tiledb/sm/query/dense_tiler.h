/**
 * @file   dense_tiler.h
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
 * This file defines class DenseTiler.
 */

#ifndef TILEDB_DENSE_TILER_H
#define TILEDB_DENSE_TILER_H

#include "tiledb/common/logger.h"
#include "tiledb/common/status.h"
#include "tiledb/sm/array_schema/array_schema.h"
#include "tiledb/sm/query/query_buffer.h"
#include "tiledb/sm/subarray/subarray.h"

#include <functional>
#include <unordered_map>

using namespace tiledb::common;

namespace tiledb {
namespace sm {

/**
 * Creates dense tiles from the input buffers for a particular
 * array schema and subarray. Used in dense writes.
 *
 * @tparam T The array domain datatype.
 */
template <class T>
class DenseTiler {
 public:
  /* ********************************* */
  /*       PUBLIC DATA STRUCTURES      */
  /* ********************************* */

  /**
   * Contains information on how the cell copy from a buffer (corresponding
   * to elements gives for the input subarray) to the
   * tile will be carried out. The copy algorithm proceeds by starting at a
   * subarray position and a tile position, and then entering a loop of a
   * fixed number of copy iterations determines by a vector of ranges
   * (which implement a ND for loop). In each copy iteration, there is a
   * fixed number of elements to be copied from the subarray (i.e., the
   * buffers) to the tile,
   * and a fixed number of elements to be skipped (i.e., a stride) inside
   * the subarray and the tile, depending on which dimension index changes
   * in the copy loop over the dimensions ranges.
   */
  struct CopyPlan {
    /**
     * Number of elements to copy from the buffer to the tile in each
     *  copy iteration.
     */
    uint64_t copy_el_;
    /**
     * This vector (one range per dimension) determines the ND copy
     * iteration loop.
     */
    std::vector<std::array<uint64_t, 2>> dim_ranges_;
    /**
     * The position of the element in the subarray the first copy iteration
     * should start from.
     */
    uint64_t sub_start_el_;
    /**
     * The number of elements to "jump" in the subarray when a dimension
     * index changes.
     */
    std::vector<uint64_t> sub_strides_el_;
    /**
     * The position of the element in the tile the first copy iteration
     * should start from.
     */
    uint64_t tile_start_el_;
    /**
     * The number of elements to "jump" in the tile when a dimension
     * index changes.
     */
    std::vector<uint64_t> tile_strides_el_;

    /** Constructor. */
    CopyPlan() = default;

    /** Constructor. */
    CopyPlan(
        uint64_t copy_el,
        const std::vector<std::array<uint64_t, 2>>& dim_ranges,
        uint64_t sub_start_el,
        const std::vector<uint64_t>& sub_strides_el,
        uint64_t tile_start_el,
        const std::vector<uint64_t>& tile_strides_el)
        : copy_el_(copy_el)
        , dim_ranges_(dim_ranges)
        , sub_start_el_(sub_start_el)
        , sub_strides_el_(sub_strides_el)
        , tile_start_el_(tile_start_el)
        , tile_strides_el_(tile_strides_el) {
    }
  };

  /* ********************************* */
  /*     CONSTRUCTORS & DESTRUCTORS    */
  /* ********************************* */

  /**
   * Constructor.
   *
   * @note It is assumed that `buffers` contains correct attributes
   *     complying with the array_schema (which can be retrieved
   *     from `subarray`). Otherwise, an assertion is raised.
   */
  DenseTiler(
      const std::unordered_map<std::string, QueryBuffer>& buffers,
      const Subarray* subarray);

  /** Destructor. */
  ~DenseTiler();

  /* ********************************* */
  /*                 API               */
  /* ********************************* */

  /** Computes and returns the copy plan for the give tile id. */
  const CopyPlan copy_plan(uint64_t id) const;

  /**
   * Retrieves the fixed-sized tile with the input id and for the input
   * attribute.
   *
   * @param id The id of the tile within the subarray to be retrieved.
   *     The id is serialied in the tile order of the array domain.
   * @param name The name of the attribute.
   * @param tile The tile to be retrieved. This needs to
   *     be preallocated and initialized before passed to the function.
   * @return Status
   */
  Status get_tile(uint64_t id, const std::string& name, Tile* tile) const;

  /**
   * Retrieves the var-sized tile with the input id and for the input
   * attribute.
   *
   * @param id The id of the tile to be retrieved. The id is serialied in the
   *     tile order of the array domain.
   * @param name The name of the attribute.
   * @param tile_off The tile with the offsets to be retrieved. This needs to
   *     be preallocated and initialized before passed to the function.
   * @param tile_val The tile with the values to be retrieves. This needs to
   *     be preallocated and initialized before passed to the function.
   * @return Status
   */
  Status get_tile(
      uint64_t id,
      const std::string& name,
      Tile* tile_off,
      Tile* tile_var) const;

  /**
   * Returns the number of tiles to be created. This is equal
   * to the number of tiles intersecting the subarray.
   */
  uint64_t tile_num() const;

  /**
   * Returns the number of elements to "jump" in the tile when a dimension
   * index changes.
   */
  const std::vector<uint64_t>& tile_strides_el() const;

  /**
   * Returns the number of elements to "jump" in the subarray when
   * a dimension index changes.
   */
  const std::vector<uint64_t>& sub_strides_el() const;

  /** Returns the tile domain of the subarray. */
  const std::vector<uint64_t>& sub_tile_coord_offsets() const;

  /** Returns the coordinates of the first tile intersecting the subarray. */
  const std::vector<uint64_t>& first_sub_tile_coords() const;

 private:
  /* ********************************* */
  /*         PRIVATE ATTRIBUTES        */
  /* ********************************* */

  /** The array schema. */
  const ArraySchema* array_schema_;

  /** The input buffers, from which the tiles will be produced. */
  std::reference_wrapper<const std::unordered_map<std::string, QueryBuffer>>
      buffers_;

  /**
   * The subarray used in the dense write. Note that this is guaranteed to
   * be a single-range subarray.
   */
  const Subarray* subarray_;

  /**
   * The number of tiles to be created, equal to the number of tiles
   * intersecting subarray_.
   */
  uint64_t tile_num_;

  /**
   * The number of elements to "jump" in the tile when a dimension index
   * changes.
   */
  std::vector<uint64_t> tile_strides_el_;

  /**
   * The number of elements to "jump" in the subarray when
   * a dimension index changes.
   */
  std::vector<uint64_t> sub_strides_el_;

  /** The tile domain of the subarray. */
  std::vector<uint64_t> sub_tile_coord_offsets_;

  /** The coordinates of the first tile intersecting the subarray. */
  std::vector<uint64_t> first_sub_tile_coords_;

  /* ********************************* */
  /*           PRIVATE METHODS         */
  /* ********************************* */

  /**
   * Calculates the tile coordinates in the array tile domain of the
   * first tile intersecting the subarray.
   */
  void calculate_first_sub_tile_coords();

  /** Calculates the subarray tile coordinate offsets. */
  void calculate_subarray_tile_coord_offsets();

  /**
   * Calculates the tile and subarray strides. These are fixed for all
   * tiles.
   */
  void calculate_tile_and_subarray_strides();

  /** Calculates the number of tiles to be created. */
  void calculate_tile_num();

  /** Fills the input tile with the array schema fill values. */
  Status fill_tile(const std::string& name, Tile* tile) const;

  /** Initializes the input tile. */
  Status init_tile(const std::string& name, Tile* tile) const;

  /**
   * Returns the tile coordinates of the given tile id inside
   * the subarray tile domain.
   */
  std::vector<uint64_t> tile_coords_in_sub(uint64_t id) const;

  /**
   * Given a tile id serialized in the tile order of the
   * array domain within the subarray, it returns the corresponding tile
   * subarray (in global coordinates).
   */
  std::vector<std::array<T, 2>> tile_subarray(uint64_t id) const;
};

}  // namespace sm
}  // namespace tiledb

#endif  // TILEDB_DENSE_TILER_H
