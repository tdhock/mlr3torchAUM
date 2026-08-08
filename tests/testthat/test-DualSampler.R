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
    )$batches
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
    )$batches
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
    )$batches
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
    # carries the pointer to the next calling
    set.seed(1)
    pos <- c(3, 6, 8, 11)
    neg <- c(1, 2, 4, 5, 7, 9, 10, 12)
    first <- dual_batch_list(pos, neg, num_pos = 2, num_neg = 4, num_batches = 1)
    expect_equal(first$batches[[1]], c(3, 6, 1, 2, 4, 5))
    expect_equal(first$pos_ptr, 2)
    expect_equal(first$neg_ptr, 4)
    second <- dual_batch_list(first$pos_pool, first$neg_pool,
      num_pos = 2, num_neg = 4, num_batches = 1,
      pos_ptr = first$pos_ptr, neg_ptr = first$neg_ptr
    )
    expect_equal(second$batches[[1]], c(8, 11, 7, 9, 10, 12))
  })

  test_that("test batch_sampler_dual", {
    skip_on_cran()
    # Reproduces LibAUC's first epoch end to end, through the sampler:
    #
    #   uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' \
    #     --python 3.11 python
    #   from libauc.sampler import DualSampler
    #   labels = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    #   s = DualSampler(None, batch_size=6, labels=labels, shuffle=False,
    #                   sampling_rate=1/3)
    #   print(s.num_pos, s.num_neg, s.num_batches)   # 2 4 2
    #   print([int(i) for i in s])                   # 0-based first epoch:
    #   # [2, 5, 0, 1, 3, 4, 7, 10, 6, 8, 9, 11]
    set.seed(1)
    labels <- c(0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0)
    task <- mlr3::TaskClassif$new("dual",
      data.frame(
        x = seq_along(labels),
        y = factor(ifelse(labels == 1, "pos", "neg"), levels = c("neg", "pos"))
      ),
      target = "y", positive = "pos"
    )
    sampler <- batch_sampler_dual(
      batch_size = 6, sampling_rate = 1 / 3,
      shuffle = FALSE
    )(list(task = task))
    expect_equal(sampler$batch_size, 6)
    expect_equal(length(sampler$batch_list), 2)
    expect_equal(unlist(sampler$batch_list), c(3, 6, 1, 2, 4, 5, 8, 11, 7, 9, 10, 12))
    # fixes the positive count per batch
    set.seed(1)
    labels <- c(rep(1, 5), rep(0, 40))
    task <- mlr3::TaskClassif$new("imbalanced",
      data.frame(
        x = seq_along(labels),
        y = factor(ifelse(labels == 1, "pos", "neg"), levels = c("neg", "pos"))
      ),
      target = "y", positive = "pos"
    )
    sampler <- batch_sampler_dual(batch_size = 10, sampling_rate = 0.3)(list(task = task))
    positives <- which(labels == 1)
    for (batch in sampler$batch_list) {
      # int(0.3 * 10) = 3 positives in every batch, however rare they are.
      expect_equal(sum(batch %in% positives), 3)
      expect_equal(length(batch), 10)
    }
    # reproducible given a random seed
    labels <- c(rep(1, 5), rep(0, 40))
    task <- mlr3::TaskClassif$new("seeded",
      data.frame(
        x = seq_along(labels),
        y = factor(ifelse(labels == 1, "pos", "neg"), levels = c("neg", "pos"))
      ),
      target = "y", positive = "pos"
    )
    sampler1 <- batch_sampler_dual(
      batch_size = 10, sampling_rate = 0.3, random_seed = 42
    )(list(task = task))
    sampler2 <- batch_sampler_dual(
      batch_size = 10, sampling_rate = 0.3, random_seed = 42
    )(list(task = task))
    expect_equal(sampler1$batch_list, sampler2$batch_list)
    # rejects a single-class task, blocked by dual_class_indices
    task <- mlr3::TaskClassif$new("one_class",
      data.frame(x = 1:6, y = factor(rep("neg", 6), levels = c("neg", "pos"))),
      target = "y", positive = "pos"
    )
    expect_error(batch_sampler_dual(batch_size = 4)(list(task = task)))
  })

  test_that("e2e: test batch_sampler_dual on a severely imbalanced task", {
    skip_on_cran()
    # 2% positives, very imbalanced
    set.seed(1)
    n <- 500
    n_pos <- 10
    labels <- c(rep(1, n_pos), rep(0, n - n_pos))
    task <- mlr3::TaskClassif$new("imbalanced",
      data.frame(
        x = seq_len(n),
        y = factor(ifelse(labels == 1, "pos", "neg"), levels = c("neg", "pos"))
      ),
      target = "y", positive = "pos"
    )
    sampler <- batch_sampler_dual(batch_size = 32, sampling_rate = 0.1)(list(task = task))
    expect_equal(sampler$num_pos, 3) # as.integer(0.1 * 32)
    expect_equal(sampler$num_neg, 29)
    expect_equal(sampler$num_batches, 16) # max(10 %/% 3, 490 %/% 29)
    positives <- which(labels == 1)
    counts <- sapply(sampler$batch_list, function(batch) sum(batch %in% positives))
    expect_true(all(counts == 3))
    expect_true(all(lengths(sampler$batch_list) == 32))
    expect_equal(sum(counts), 48) # 16 batches * 3 positives
    expect_equal(length(unique(unlist(sampler$batch_list)[
      unlist(sampler$batch_list) %in% positives
    ])), n_pos) # get all positive samples
    # Contrast with plain random
    set.seed(1)
    random_batches <- split(sample(n), ceiling(seq_len(n) / 32))
    random_counts <- sapply(random_batches, function(batch) sum(batch %in% positives))
    expect_gt(sum(random_counts == 0), 0)
    expect_equal(sum(counts == 0), 0)
  })
}
