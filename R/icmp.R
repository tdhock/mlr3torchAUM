mcp_curve<-function(y_true,y_score,labels=NULL,abs_tolerance=1e-8) {
  y_true_vec<-as.vector(y_true)
  y_score_t<-.ensure_score_tensor(y_score)
  .validate_inputs(y_true_vec,y_score_t,abs_tolerance)
  mapped<-.map_class_labels(y_true_vec,y_score_t,labels)
  y_true_score<-torch::torch_eye(mapped$y_true_size)[mapped$y_true_int_encoded,]
  result<-.get_y_values(y_true_vec,y_true_score,y_score_t)
  curve_y<-result$curve_y
  n<-curve_y$shape[1]
  curve_x<-torch::torch_linspace(0,1,steps=n)
  list(curve_x=curve_x,curve_y=curve_y)
}

imcp_curve<-function(y_true,y_score,labels=NULL,abs_tolerance=1e-8) {
  y_true_vec<-as.vector(y_true)
  y_score_t<-.ensure_score_tensor(y_score)
  .validate_inputs(y_true_vec,y_score_t,abs_tolerance)
  mapped<-.map_class_labels(y_true_vec,y_score_t,labels)
  y_true_size<-mapped$y_true_size
  y_true_int_encoded<-mapped$y_true_int_encoded
  y_true_score<-torch::torch_eye(y_true_size)[y_true_int_encoded,]
  result<-.get_y_values(y_true_vec,y_true_score,y_score_t)
  curve_y<-result$curve_y
  sort_indices<-result$sort_indices
  class_widths<-.get_class_widths(y_true_score,y_true_size)
  class_widths_per_sample<-class_widths[y_true_int_encoded]
  curve_x<-class_widths_per_sample[sort_indices]
  curve_x<-torch::torch_cumsum(curve_x,dim=1)-(curve_x/2)
  n_y<-curve_y$shape[1]
  curve_x<-torch::torch_cat(list(torch::torch_zeros(1),curve_x,torch::torch_ones(1)))
  curve_y<-torch::torch_cat(list(
    curve_y[1]$unsqueeze(1),
    curve_y,
    curve_y[n_y]$unsqueeze(1)
  ))
  list(curve_x=curve_x,curve_y=curve_y)
}

mcp_score<-function(y_true,y_score,labels=NULL,abs_tolerance=1e-8) {
  curves<-mcp_curve(y_true,y_score,labels=labels,abs_tolerance=abs_tolerance)
  as.numeric(.torch_trapezoid(curves$curve_y,curves$curve_x))
}

imcp_score<-function(y_true,y_score,labels=NULL,abs_tolerance=1e-8) {
  curves<-imcp_curve(y_true,y_score,labels=labels,abs_tolerance=abs_tolerance)
  as.numeric(.torch_trapezoid(curves$curve_y,curves$curve_x))
}

.map_class_labels<-function(y_true,y_score,labels) {
  unique_classes<-sort(unique(y_true))
  y_true_size<-length(unique_classes)
  class_mapper<-setNames(seq_len(y_true_size),unique_classes)
  y_true_int_encoded<-unname(class_mapper[as.character(y_true)])
  n_cols<-y_score$shape[2]
  if(y_true_size!=n_cols) {
    if(is.null(labels)) {
      stop("Class labels not given!")
    }
    if(length(labels)!=n_cols) {
      stop("Number of class labels not equal to the number of columns in 'y_score'")
    }
    if(!all(unique_classes%in%labels)) {
      stop(
        "Class labels from y_true are not a subset of given list of labels. ",
        "Check if values and types of given labels and y_true match."
      )
    }
    unique_classes<-sort(labels)
    y_true_size<-length(unique_classes)
    class_mapper<-setNames(seq_len(y_true_size),unique_classes)
    y_true_int_encoded<-unname(class_mapper[as.character(y_true)])
  }
  list(
    class_mapper=class_mapper,
    y_true_size=y_true_size,
    y_true_int_encoded=as.integer(y_true_int_encoded)
  )
}

.get_y_values<-function(y_true,y_true_score,y_score) {
  curve_y<-(y_true_score-torch::torch_sqrt(y_score))$pow(2)
  curve_y<-torch::torch_sum(curve_y,dim=2)
  curve_y<-torch::torch_sqrt(curve_y)/sqrt(2)
  curve_y<-1-curve_y
  curve_y_vec<-as.numeric(curve_y)
  sort_indices<-order(curve_y_vec,y_true)
  curve_y<-curve_y[sort_indices]
  list(curve_y=curve_y,sort_indices=sort_indices)
}

.get_class_widths<-function(y_true_score,y_true_size) {
  class_widths<-torch::torch_sum(y_true_score,dim=1)
  1.0/(y_true_size*class_widths)
}

.torch_trapezoid<-function(y,x) {
  n<-y$shape[1]
  dx<-x[2:n]-x[1:(n-1)]
  avg_y<-(y[2:n]+y[1:(n-1)])/2.0
  torch::torch_sum(dx*avg_y)
}

.ensure_score_tensor<-function(y_score) {
  if(torch::is_torch_tensor(y_score)) y_score$to(dtype=torch::torch_float())
  else torch::torch_tensor(as.matrix(y_score),dtype=torch::torch_float())
}

.validate_inputs<-function(y_true_vec,y_score_t,abs_tolerance) {
  if(length(y_true_vec)!=y_score_t$shape[1]) {
    stop("'y_true' and 'y_score' have different number of samples")
  }
  row_sums<-y_score_t$sum(dim=2)
  if(!torch::torch_allclose(torch::torch_ones_like(row_sums),row_sums,rtol=0,atol=abs_tolerance)) {
    stop(
      "Target scores need to be probabilities, ",
      "i.e. they should sum up to 1.0 over classes"
    )
  }
}
