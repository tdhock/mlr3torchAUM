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
}
