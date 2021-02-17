/**
 * @file   query_condition.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2021 TileDB, Inc.
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
 * This file declares the C++ API for the TileDB QueryCondition object.
 */

#ifndef TILEDB_CPP_API_QUERY_CONDITION_H
#define TILEDB_CPP_API_QUERY_CONDITION_H

#include "context.h"
#include "object.h"
#include "tiledb.h"

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

namespace tiledb {

class QueryCondition {
 public:
  /* ********************************* */
  /*     CONSTRUCTORS & DESTRUCTORS    */
  /* ********************************* */

  QueryCondition(
      const Context& ctx,
      const std::string& attribute_name,
      const void* condition_value,
      uint64_t condition_value_size,
      tiledb_query_condition_op_t op) {
    tiledb_query_condition_t* qc;
    ctx.handle_error(tiledb_query_condition_alloc(
        ctx.ptr().get(),
        attribute_name.c_str(),
        condition_value,
        condition_value_size,
        op,
        &qc));
    query_condition_ = std::shared_ptr<tiledb_query_condition_t>(qc, deleter_);
  }

  QueryCondition();
  QueryCondition(const QueryCondition&) = default;
  QueryCondition(QueryCondition&&) = default;
  QueryCondition& operator=(const QueryCondition&) = default;
  QueryCondition& operator=(QueryCondition&&) = default;

  /* ********************************* */
  /*                API                */
  /* ********************************* */

  /** Returns a shared pointer to the C TileDB query condition object. */
  std::shared_ptr<tiledb_query_condition_t> ptr() const {
    return query_condition_;
  }

  /* ********************************* */
  /*          STATIC FUNCTIONS         */
  /* ********************************* */

  template <typename T>
  static QueryCondition create(
      const Context& ctx,
      const std::string& attribute_name,
      T value,
      tiledb_query_condition_op_t op) {
    QueryCondition qc(ctx, attribute_name, &value, sizeof(T), op);
    return qc;
  }

 private:
  /* ********************************* */
  /*         PRIVATE ATTRIBUTES        */
  /* ********************************* */

  /** Deleter wrapper. */
  impl::Deleter deleter_;

  /** Pointer to the TileDB C query object. */
  std::shared_ptr<tiledb_query_condition_t> query_condition_;
};

}  // namespace tiledb

#endif  // TILEDB_CPP_API_QUERY_CONDITION_H
