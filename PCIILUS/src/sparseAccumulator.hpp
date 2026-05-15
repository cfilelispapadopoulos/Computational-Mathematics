#ifndef sparseAccumulator_HPP
#define sparseAccumulator_HPP
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <utility>

// DISCLAIMER
// The use of the code is regulated by the following copyright agreement.
//
// PCIILUS software is freely available for scientific (non-commercial) use.
// 1. The code can be used only for the purpose of internal research, excluding any commercial use of the PCIILUS
//    software as such or as a part of a software product. Users who want to integrate PCIILUS sofware or parts of
//    it into commercial products require a license agreement.
// 2. PCIILUS is provided "as is" and for the purpose described at the previous point only. In no circumstances can
//    neither the authors nor their institutions be held liable for any deficiency, fault or other mishappening
//    with regard to the use or performance of PCIILUS.
// 3. All scientific publications, for which PCIILUS software has been used, shall mention its usage and refer to
//    the publication [1] in the References section below.
//
//
// References
// [1] C. K. Filelis - Papadopoulos (2026). Parallel Incomplete LU Factorization, Submitted.

// sparseAccumulator - Class that implements a sparse accumulator used for sparse matrix operations
//
// Author: Christos K. Papadopoulos Filelis
//         Assistant Professor
//         Democritus University of Thrace
//         Department of Electrical and Computer Engineering
//         Xanthi, Greece, GR 67100
//         email: cpapad@ee.duth.gr
//
// ---------------------- Arguments -------------------------------------------------------------------------
// MEMBERS
// NAME             TYPE                DESCRIPTION
// v                (S*)                Vector retaining the nonzero values (float, double, long double,...)
// i                (R*)                Vector retaining the nonzero indices (int8, int16,...)
// last             (P)                 Index to the last position
// tlast            (P)                 Temporary Index to the last position
// n                (P)                 Size of sparse row or column
// nnz              (P)                 Number of nonzero elements
//
// METHODS
// sparseAccumulator()                              Default constructor
// sparseAccumulator(P n)                           Second contructor that allocates the internal structure
// sparseAccumulator(sparseAccumulator &A)          Copy constructor
// resize(P n)                                      Resize sparse accumulator
// clear()                                          Deallocates space
// rewind()                                         Rewinds temporary pointer to its original position
// push(R& i_, S& v_)                               Push element into accumulator
// delete_last()                                    Delete last element of the accumulator
// S& top(R& i_)                                    Return last element of the accumulator
// S& next(R& i_)                                   Next element in the accumulator
// ~sparseAccumulator()                             Default deconstructor

template <typename P, typename R, typename S>
class sparseAccumulator
{
public:
    std::vector<S> v;
    std::vector<R> i;
    P n, last, tlast, nnz;

    // sparseAccumulator - Default constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    sparseAccumulator()
    {
        last = P(-2);
        tlast = P(-2);
        n = P(0);
        nnz = P(0);
    };

    // sparseAccumulator - Secondary constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // n_               (P)                 Size of the sparse line or column

    sparseAccumulator(P n_)
    {
        resize(n_);
    };

    // sparseAccumulator - Copy constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // A                (sparseAccumulator) Sparse accumulator

    sparseAccumulator(sparseAccumulator &A)
    {
        n = A.n;
        last = A.last;
        tlast = A.tlast;
        nnz = A.last;
        v = A.v;
        i = A.i;
    };

    // clear - Function that clears class
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void clear()
    {
        last = -2;
        tlast = -2;
        n = 0;
        nnz = 0;
        i = std::vector<R>();
        v = std::vector<S>();
    }

    // push - Push an element into the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Position of element to be pushed
    // v_               (S)                 Value of element to be pushed

    inline void push(R &i_, S &v_)
    {
        if (isempty(i_))
        {
            v[i_] = v_;
            i[i_] = last;
            last = i_;
            nnz++;
        }
        else
            v[i_] += v_;
    }

    // delete_last - Pop an element out of the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void delete_last()
    {
        R i_ = last;
        last = i[i_];
        v[i_] = 0.0;
        i[i_] = -1;
        nnz--;
    }

    // top - Return last element of the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Index of last element
    // (rval)           (S)                 Value of last element

    S &top(R &i_)
    {
        i_ = last;
        return v[last];
    }

    // isempty - Check if position is empty
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Check position of element
    // (rval)           (bool)              True if empty

    inline bool isempty(R &i_)
    {
        return (i[i_] == -1);
    }

    // empty - Empty the sparse accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void empty()
    {
        for (int j = 0; j < nnz; j++)
        {
            R i_ = last;
            last = i[i_];
            v[i_] = 0.0;
            i[i_] = -1;
        }
        nnz = 0;
    }

    // resize - Secondary constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // n_               (P)                 Size of the sparse line or column

    void resize(P n_)
    {
        n = n_;
        last = P(-2);
        tlast = P(-2);
        nnz = P(0);
        v.resize(n, S(0.0));
        i.resize(n, R(-1));
    };

    // rewind - Rewind the temporary pointer to access the list
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void rewind()
    {
        tlast = last;
    };

    // next - Return next element on the list
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Index of the nonzero value
    //
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (rval)           (S)                 Value of the next nonzero

    inline S &next(R &i_)
    {
        // Reverse iterator
        i_ = tlast;
        tlast = i[i_];
        return v[i_];
    };

    // sparseAccumulator - Deconstructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    ~sparseAccumulator() {

    };
};


template <typename P, typename R>
class sparseAccumulatorSymbolic
{
public:
    std::vector<R> i, o;
    P n, last, tlast, nnz;

    // sparseAccumulator - Default constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    sparseAccumulatorSymbolic()
    {
        last = P(-2);
        tlast = P(-2);
        n = P(0);
        nnz = P(0);
    };

    // sparseAccumulator - Secondary constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // n_               (P)                 Size of the sparse line or column

    sparseAccumulatorSymbolic(P n_)
    {
        resize(n_);
    };

    // sparseAccumulator - Copy constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // A                (sparseAccumulator) Sparse accumulator

    sparseAccumulatorSymbolic(sparseAccumulatorSymbolic &A)
    {
        n = A.n;
        last = A.last;
        tlast = A.tlast;
        nnz = A.nnz;
        i = A.i;
        o = A.o;
    };

    // clear - Function that clears class
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void clear()
    {
        last = -2;
        tlast = -2;
        n = 0;
        nnz = 0;
        i = std::vector<R>();
        o = std::vector<R>();
    }

    // push - Push an element into the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Position of element to be pushed

    inline void push(R &i_)
    {
        if (isempty(i_))
        {
            i[i_] = last;
            o[i_] = nnz;
            last = i_;
            nnz++;
        }
    }

    // delete_last - Pop an element out of the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void delete_last()
    {
        R i_ = last;
        last = i[i_];
        i[i_] = -1;
        o[i_] = -1;
        nnz--;
    }

    // top - Return last element of the accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (rval)           (R)                 Index of the element at the top

    R &top()
    {
        return last;
    }

    R &top(int &idx)
    {
        idx = o[last];
        return last;
    }

    // isempty - Check if position is empty
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // i_               (R)                 Position of element
    //
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (rval)           (bool)              True if empty

    inline bool isempty(R &i_)
    {
        return (i[i_] == -1);
    }

    // empty - Empty the sparse accumulator
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void empty()
    {
        for (int j = 0; j < nnz; j++)
        {
            R i_ = last;
            last = i[i_];
            i[i_] = -1;
            o[i_] = -1;
        }
        nnz = 0;
    }

    // resize - Secondary constructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // n_               (P)                 Size of the sparse line or column

    void resize(P n_)
    {
        n = n_;
        last = P(-2);
        tlast = P(-2);
        nnz = P(0);
        i.resize(n, R(-1));
        o.resize(n, R(-1));
    };

    // rewind - Rewind the temporary pointer to access the list
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    void rewind()
    {
        tlast = last;
    };

    // next - Return next element on the list
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    //
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (rval)           (R)                 Index of the next nonzero value

    inline void next(R &i_)
    {
        // Reverse iterator
        i_ = tlast;
        tlast = i[i_];
    };

    inline void next(R &i_, R &o_)
    {
        // Reverse iterator
        i_ = tlast;
        o_ = o[i_];
        tlast = i[i_];
    };

    // sparseAccumulator - Deconstructor
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION

    ~sparseAccumulatorSymbolic() {

    };
};

// sparseAccumulatorSymbolicHash - Class that implements a hash-based symbolic sparse accumulator
//                                 used for sparse matrix operations
//
// Author: Christos K. Papadopoulos Filelis
//         Assistant Professor
//         Democritus University of Thrace
//         Department of Electrical and Computer Engineering
//         Xanthi, Greece, GR 67100
//         email: cpapad@ee.duth.gr
//
// DESCRIPTION
// This class implements a symbolic sparse accumulator (SPA) using a flat open-addressing
// hash table with linear probing. The accumulator stores only the active symbolic indices
// of a sparse row or column, avoiding the allocation of dense marker arrays of size n.
//
// The structure maintains an active index list for traversal and ordering, while membership
// and position queries are handled through the hash table. Logical clearing is performed
// using generation stamps, allowing the accumulator to be reset without explicitly clearing
// the full hash table.
//
// The hash table uses a SplitMix64-style integer mixing function and power-of-two table
// sizes, so bucket selection is performed using a bit mask instead of modulo division.
//
// Auxiliary memory scales with the number of active entries rather than the full row/column
// dimension, i.e. O(nnz) instead of O(n), subject to the selected hash load factor.
//
// ---------------------- Arguments -------------------------------------------------------------------------
// TEMPLATE PARAMETERS
// NAME             TYPE                DESCRIPTION
// P                integral type       Integer type used for sizes, counters, positions, and ordering
// R                integral type       Integer type used for sparse row/column indices
//
// ---------------------- Members ---------------------------------------------------------------------------
// MEMBERS
// NAME             TYPE                DESCRIPTION
// active_          std::vector<R>      List of active sparse indices in insertion order
// table_           std::vector<Slot>   Flat open-addressing hash table storing keys, positions, and stamps
// n_               P                   Size of the sparse row or column, i.e. valid index range [0,n_)
// nnz_             P                   Number of active symbolic entries currently stored
// cursor_          P                   Temporary reverse-iteration cursor used by rewind()/next()
// stamp_           uint64_t            Current generation stamp used for logical clearing
//
// ---------------------- Internal Structures ---------------------------------------------------------------
// STRUCT
// NAME             TYPE                DESCRIPTION
// Slot             struct              Hash-table entry
// Slot::key        R                   Sparse matrix index stored in the slot
// Slot::pos        P                   Position/order associated with the sparse index
// Slot::stamp      uint64_t            Generation stamp indicating whether the slot is active
//
// ---------------------- Methods ---------------------------------------------------------------------------
// METHODS
// sparseAccumulatorSymbolicHash()                         Default constructor
// sparseAccumulatorSymbolicHash(P n, size_t expected_nnz) Constructor that allocates the internal hash table
// sparseAccumulatorSymbolicHash(const sparseAccumulatorSymbolicHash &A)
//                                                         Copy constructor
// sparseAccumulatorSymbolicHash(sparseAccumulatorSymbolicHash &&A)
//                                                         Move constructor
// operator=(const sparseAccumulatorSymbolicHash &A)       Copy assignment operator
// operator=(sparseAccumulatorSymbolicHash &&A)            Move assignment operator
// resize(P n, size_t expected_nnz)                        Resize accumulator and initialize hash capacity
// clear()                                                 Deallocate internal storage and reset state
// size()                                                  Return number of active symbolic entries
// capacity_active()                                       Return capacity of active index buffer
// empty_accumulator()                                     Return true if accumulator contains no active entries
// isempty(R idx)                                          Return true if index idx is not present
// push(R idx)                                             Insert symbolic index idx if it is not already present
// delete_last()                                           Remove the most recently inserted active index
// top()                                                   Return the most recently inserted active index
// top(P &idx_pos)                                         Return the most recent index and its active-list position
// empty()                                                 Logically clear accumulator using generation stamping
// rewind()                                                Reset reverse-iteration cursor
// next(R &idx)                                            Return next active index in reverse insertion order
// next(R &idx, P &ord)                                    Return next active index and its active-list order
// set_order(R idx, P ord)                                 Assign ordering/position ord to active index idx
// order_of(R idx)                                         Return ordering/position associated with active index idx
// ~sparseAccumulatorSymbolicHash()                        Default destructor

template <typename P, typename R>
class sparseAccumulatorSymbolicHash
{
    static_assert(std::is_integral_v<P>, "P must be an integral type");
    static_assert(std::is_integral_v<R>, "R must be an integral type");

private:
    static constexpr R invalid_index = R(-1);

    struct Slot
    {
        R key = invalid_index;
        P pos = P(-1);
        std::uint64_t stamp = 0;
    };

    std::vector<R> active_;
    std::vector<Slot> table_;

    P n_ = 0;
    P nnz_ = 0;
    P cursor_ = 0;

    std::uint64_t stamp_ = 1;

    // Load factor for the Hash based SPA
    static constexpr double max_load_factor_ = 0.50;

    // next_pow2 - Computes next power of 2
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // x                (size_t)            Input
    static inline std::size_t next_pow2(std::size_t x)
    {
        if (x <= 1)
            return 1;

        --x;
        for (std::size_t s = 1; s < sizeof(std::size_t) * 8; s <<= 1)
            x |= x >> s;

        return x + 1;
    }

    // mix - SplitMix64 finalizer 
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // x                (size_t)            Input.   
    static inline std::size_t mix(std::size_t x)
    {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    // mask - Masking function
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION      
    inline std::size_t mask() const noexcept
    {
        return table_.size() - 1;
    }

    // active_slot - Checks if slot is active
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // s                (Slot)              Input.
    inline bool active_slot(const Slot &s) const noexcept
    {
        return s.stamp == stamp_;
    }

    // init_table - Initilizes the table.
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // min_capacity     (size_t)            The minimum capacity for the table.
    void init_table(std::size_t min_capacity)
    {
        std::size_t cap = next_pow2(min_capacity < 8 ? 8 : min_capacity);
        table_.assign(cap, Slot{});
    }

    // maybe_rehash_for_insert - Checks insertion and rehashes table.
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION    
    void maybe_rehash_for_insert()
    {
        if (table_.empty())
        {
            init_table(8);
            return;
        }

        const double lf =
            static_cast<double>(nnz_ + 1) /
            static_cast<double>(table_.size());

        if (lf > max_load_factor_)
            rehash(table_.size() * 2);
    }

    // find_slot_or_empty - Finds a slot or an empty one
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION  
    // key              (R)                 Key to search against.      
    std::size_t find_slot_or_empty(R key) const noexcept
    {
        std::size_t h = mix(static_cast<std::size_t>(key)) & mask();

        while (true)
        {
            const Slot &s = table_[h];

            if (!active_slot(s) || s.key == key)
                return h;

            h = (h + 1) & mask();
        }
    }

    // find_existing_slot - Finds existing slot based on a key.
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION  
    // key              (R)                 Key to search against.    
    std::size_t find_existing_slot(R key) const noexcept
    {
        if (table_.empty())
            return table_.size();

        std::size_t h = mix(static_cast<std::size_t>(key)) & mask();

        while (true)
        {
            const Slot &s = table_[h];

            if (!active_slot(s))
                return table_.size();

            if (s.key == key)
                return h;

            h = (h + 1) & mask();
        }
    }

    // rehash - Rehashes table in case of new cap
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION  
    // new_cap          (size_t)            The new cap to expand table
    void rehash(std::size_t new_cap)
    {
        std::vector<R> old_active = active_;

        init_table(new_cap);

        for (P k = 0; k < nnz_; ++k)
        {
            R idx = old_active[k];
            std::size_t h = find_slot_or_empty(idx);

            table_[h].key = idx;
            table_[h].pos = k;
            table_[h].stamp = stamp_;
        }

        active_.swap(old_active);
    }

    // reset_all_stamps - Resets all stamps
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION  
    void reset_all_stamps()
    {
        for (Slot &s : table_)
            s.stamp = 0;

        stamp_ = 1;
    }

public:
    sparseAccumulatorSymbolicHash() = default;

    explicit sparseAccumulatorSymbolicHash(P n, std::size_t expected_nnz = 16)
    {
        resize(n, expected_nnz);
    }

    sparseAccumulatorSymbolicHash(const sparseAccumulatorSymbolicHash &) = default;
    sparseAccumulatorSymbolicHash(sparseAccumulatorSymbolicHash &&) noexcept = default;
    sparseAccumulatorSymbolicHash &operator=(const sparseAccumulatorSymbolicHash &) = default;
    sparseAccumulatorSymbolicHash &operator=(sparseAccumulatorSymbolicHash &&) noexcept = default;
    ~sparseAccumulatorSymbolicHash() = default;

    // resize - Resizes active vector
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // n                (P)                 Real size
    // expected_nnz     (size_t)            Expected size of nonzero elements in vector of size n
    void resize(P n, std::size_t expected_nnz = 16)
    {
        n_ = n;
        nnz_ = 0;
        cursor_ = 0;
        stamp_ = 1;

        active_.clear();
        active_.reserve(expected_nnz);

        const std::size_t cap =
            static_cast<std::size_t>(
                static_cast<double>(expected_nnz) / max_load_factor_ + 1.0);

        init_table(cap);
    }

    // clear - Clear SPA
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION 
    void clear()
    {
        n_ = 0;
        nnz_ = 0;
        cursor_ = 0;
        stamp_ = 1;

        active_.clear();
        table_.clear();
    }

    // size - Return number of nonzero elements
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    inline P size() const noexcept
    {
        return nnz_;
    }

    // capacity_active - Capacity of active indices
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION    
    inline P capacity_active() const noexcept
    {
        return static_cast<P>(active_.capacity());
    }

    // empty_accumulator - Check if the accumulator is empty
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION        
    inline bool empty_accumulator() const noexcept
    {
        return nnz_ == 0;
    }

    // isempty - Check if index is empty
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.
    inline bool isempty(R idx) const noexcept
    {
        return find_existing_slot(idx) == table_.size();
    }

    // push - Pushes new value at a specific index
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.    
    inline void push(R idx)
    {
        if (idx < 0 || idx >= static_cast<R>(n_))
            throw std::out_of_range("sparseAccumulatorSymbolicHash::push index out of bounds");

        maybe_rehash_for_insert();

        std::size_t h = find_slot_or_empty(idx);

        if (active_slot(table_[h]))
            return;

        table_[h].key = idx;
        table_[h].pos = nnz_;
        table_[h].stamp = stamp_;

        active_.push_back(idx);
        ++nnz_;
    }

    // delete_last - Deletes last element
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION    
    void delete_last()
    {
        if (nnz_ == 0)
            throw std::runtime_error("sparseAccumulatorSymbolicHash::delete_last on empty accumulator");

        R idx = active_.back();
        active_.pop_back();
        --nnz_;

        std::size_t h = find_existing_slot(idx);
        if (h != table_.size())
        {
            table_[h].stamp = 0;
            table_[h].pos = P(-1);
        }

        if (cursor_ > nnz_)
            cursor_ = nnz_;
    }

    // top - Outputs last element
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // 
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (lval)           (R)                 Element.
    R top() const
    {
        if (nnz_ == 0)
            throw std::runtime_error("sparseAccumulatorSymbolicHash::top on empty accumulator");

        return active_[nnz_ - 1];
    }

    // top - Outputs element in specific position
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx_pos          (R)                 Index.
    //
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (lval)           (R)                 Element.    
    R top(P &idx_pos) const
    {
        if (nnz_ == 0)
            throw std::runtime_error("sparseAccumulatorSymbolicHash::top on empty accumulator");

        idx_pos = nnz_ - 1;
        return active_[nnz_ - 1];
    }

    // empty - Empty the SPA
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION      
    void empty()
    {
        active_.clear();
        nnz_ = 0;
        cursor_ = 0;

        ++stamp_;

        if (stamp_ == 0)
            reset_all_stamps();
    }

    // rewind - Rewind cursor to the beginning
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION         
    void rewind()
    {
        cursor_ = nnz_;
    }

    // next - Give next elements position
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.           
    inline void next(R &idx)
    {
        if (cursor_ == 0)
        {
            idx = invalid_index;
            return;
        }

        --cursor_;
        idx = active_[cursor_];
    }

    // next - Give next elements position and count
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.
    // ord              (R)                 Order.    
    inline void next(R &idx, P &ord)
    {
        if (cursor_ == 0)
        {
            idx = invalid_index;
            ord = P(-1);
            return;
        }

        --cursor_;
        idx = active_[cursor_];
        ord = cursor_;
    }

    // set_order - Set the order of nnz at position idx
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.
    // ord              (R)                 Order.     
    void set_order(R idx, P ord)
    {
        std::size_t h = find_existing_slot(idx);

        if (h == table_.size())
            throw std::runtime_error("sparseAccumulatorSymbolicHash::set_order index not present");

        table_[h].pos = ord;
    }

    // order_of - Get the order of nnz at position idx
    //
    // ---------------------- Arguments -------------------------------------------------------------------------
    // INPUTS
    // NAME             TYPE                DESCRIPTION
    // idx              (R)                 Index.
    //
    // OUTPUTS
    // NAME             TYPE                DESCRIPTION
    // (lval)           (P)                 Order of element.      
    P order_of(R idx) const
    {
        std::size_t h = find_existing_slot(idx);

        if (h == table_.size())
            throw std::runtime_error("sparseAccumulatorSymbolicHash::order_of index not present");

        return table_[h].pos;
    }
};

#endif
