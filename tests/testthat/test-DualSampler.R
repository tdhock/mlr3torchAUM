if (torch::torch_is_installed() && requireNamespace("mlr3torch")) {
  test_that("test dual_num_pos", {
    skip_on_cran()
    expect_equal(dual_num_pos(6, 1 / 3), 2)
    expect_equal(dual_num_pos(8, 0.05), 1)
    expect_equal(dual_num_pos(6, NULL, num_pos = 3), 3)
    # sampling_rate and num_pos cannot be given at same time
    expect_error(dual_num_pos(6, 0.5, num_pos = 3))
  })

  test_that("test dual_num_batches", {
    skip_on_cran()
    expect_equal(dual_num_batches(4, 8, 2, 4), 2)
    expect_equal(dual_num_batches(2, 8, 2, 3), 2) # 2 neg sample unused
  })

  test_that("test dual_class_indices", {
    skip_on_cran()
    labels <- c(0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0)
    indices <- dual_class_indices(labels)
    expect_equal(indices$pos, c(3, 6, 8, 11))
    expect_equal(indices$neg, c(1, 2, 4, 5, 7, 9, 10, 12))
    # must have both labels
    expect_error(dual_class_indices(c(0, 0, 0, 0)))
    expect_error(dual_class_indices(c(1, 1, 1)))
    # don't support multiclass
    expect_error(dual_class_indices(c(0, 1, 2)))
  })

  test_that("test dual_batch_list", {
    skip_on_cran()
    # without wrap
    # uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' --python 3.11 python
    #   from libauc.sampler import DualSampler
    #   labels = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    #   s = DualSampler(None, batch_size=6, labels=labels, shuffle=False,
    #                   sampling_rate=1/3)
    #   print(s.num_pos, s.num_neg, s.num_batches)
    #   print([int(i) for i in s])
    #   # [2, 5, 0, 1, 3, 4, 7, 10, 6, 8, 9, 11]
    labels <- c(0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0)
    indices <- dual_class_indices(labels)
    batches <- dual_batch_list(indices$pos, indices$neg,
      num_pos = 2, num_neg = 4, num_batches = 2
    )
    expect_equal(length(batches), 2)
    expect_equal(lengths(batches), c(6, 6))
    expect_equal(unlist(batches), c(3, 6, 1, 2, 4, 5, 8, 11, 7, 9, 10, 12)) # 1-based
    for (batch in batches) {
      expect_true(all(batch[1:2] %in% indices$pos))
      expect_true(all(batch[3:6] %in% indices$neg))
    }
    # with wrap
    #   from libauc.sampler import DualSampler
    #   labels = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #   s = DualSampler(None, batch_size=5, labels=labels, shuffle=False,
    #                   sampling_rate=0.4)
    #   print(s.num_pos, s.num_neg, s.num_batches) # 2 3 2
    set.seed(1)
    labels <- c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    indices <- dual_class_indices(labels)
    expect_equal(indices$pos, c(1, 6))
    batches <- dual_batch_list(indices$pos, indices$neg,
      num_pos = 2, num_neg = 3, num_batches = 2
    )
    expect_equal(length(batches), 2)
    expect_equal(lengths(batches), c(5, 5))
    drawn_pos <- list()
    drawn_neg <- list()
    for (batch in batches) {
      in_pos <- batch[batch %in% indices$pos]
      in_neg <- batch[batch %in% indices$neg]
      expect_equal(length(in_pos), 2)
      expect_equal(length(in_neg), 3)
      drawn_pos[[length(drawn_pos) + 1]] <- in_pos
      drawn_neg[[length(drawn_neg) + 1]] <- in_neg
    }
    expect_equal(length(unlist(drawn_pos)), 4)
    expect_equal(length(unique(unlist(drawn_pos))), 2) # oversampling
    expect_equal(length(unlist(drawn_neg)), 6)
    expect_equal(length(unique(unlist(drawn_neg))), 6)
    expect_true(all(unlist(batches) %in% seq_along(labels)))
    # empty list
    set.seed(1)
    labels <- c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    indices <- dual_class_indices(labels)
    batches <- dual_batch_list(indices$pos, indices$neg,
      num_pos = 2, num_neg = 3, num_batches = 0
    )
    expect_equal(length(batches), 0)
  })

  test_that("test dual_take", {
    skip_on_cran()
    set.seed(1)
    pool <- c(1, 2)
    taken <- dual_take(pool, ptr = 0, need = 5)$taken
    expect_equal(length(taken), 5)
    expect_true(all(taken %in% pool))
    # keeps ptr below the pool length
    set.seed(1)
    pool <- c(3, 6, 8, 11)
    taken <- dual_take(pool, ptr = 2, need = 2)
    expect_equal(taken$taken, c(8, 11))
    expect_equal(taken$ptr, 0)
    for (ptr in 0:3) {
      expect_lt(dual_take(pool, ptr = ptr, need = 2)$ptr, length(pool))
    }
    # tiles a pool several times over
    set.seed(1)
    pool <- c(1, 2)
    # need = 9 spans the two-element pool more than four times
    taken <- dual_take(pool, ptr = 0, need = 9)$taken
    expect_equal(length(taken), 9)
    expect_true(all(taken %in% pool))
    # A pool of 3 with need = 7 and a non-zero start pointer
    bigger <- dual_take(c(1, 2, 3), ptr = 1, need = 7)
    expect_equal(length(bigger$taken), 7)
    expect_true(all(bigger$taken %in% c(1, 2, 3)))
    expect_equal(bigger$ptr, 2)
    # with an empty tail
    set.seed(1)
    taken <- dual_take(c(1, 2), ptr = 2, need = 3)$taken
    expect_equal(length(taken), 3)
    expect_true(all(taken %in% c(1, 2)))
  })
}
