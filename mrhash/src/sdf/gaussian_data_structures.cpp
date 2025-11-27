#include "gaussian_data_structures.cuh"

namespace cupanutils {
  namespace cugeoutils {
    template <typename T>
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::GaussianContainer(
      std::string gs_optimization_param_path) :
      gs_model_(gs::param::read_optim_params_from_json(gs_optimization_param_path), ".") {
      CUDA_CHECK(cudaMalloc((void**) &d_qtree_nodes_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_num_qtree_nodes_, sizeof(size_t)));
      CUDA_CHECK(cudaMalloc((void**) &d_num_valid_qtree_nodes_, sizeof(uint)));
      CUDA_CHECK(cudaMalloc((void**) &d_positions_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_colors_, sizeof(gs::CUDANode) * 1000000));
      CUDA_CHECK(cudaMalloc((void**) &d_scales_, sizeof(gs::CUDANode) * 1000000));
    }

    template <typename T>
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::~GaussianContainer() {
      CUDA_CHECK(cudaFree(d_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_num_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_num_valid_qtree_nodes_));
      CUDA_CHECK(cudaFree(d_positions_));
      CUDA_CHECK(cudaFree(d_colors_));
      CUDA_CHECK(cudaFree(d_scales_));
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::setupGSCamera(Camera& camera) {
      /* SETUP GS Camera */
      Eigen::Matrix4f T_SW     = Eigen::Isometry3f(CUDA2Eig(camera.camInWorld())).inverse().matrix();
      torch::Tensor W2C_matrix = torch::from_blob(T_SW.data(), {4, 4}, torch::kFloat).clone().to(torch::kCUDA, true);
      torch::Tensor proj_matrix =
        gs::getProjectionMatrix(camera.cols(), camera.rows(), camera.fx(), camera.fy(), camera.cx(), camera.cy())
          .to(torch::kCUDA, true);

      cur_gs_cam_.height           = camera.rows();
      cur_gs_cam_.width            = camera.cols();
      cur_gs_cam_.T_W2C            = W2C_matrix;
      cur_gs_cam_.fov_x            = camera.hfov();
      cur_gs_cam_.fov_y            = camera.vfov();
      cur_gs_cam_.T_W2C            = W2C_matrix;
      cur_gs_cam_.full_proj_matrix = W2C_matrix.mm(proj_matrix);
      cur_gs_cam_.cam_center       = W2C_matrix.inverse()[3].slice(0, 0, 3);
      gs_cam_list_.push_back(cur_gs_cam_);
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::extractNodesQTree(
      Camera& camera,
      VoxelContainer<T>& container,
      cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
      cupanutils::cugeoutils::CUDAMatrixf& depth_img) {
  gs::CUDAQTree qtree(gs_model_.optimParams.qtree_thresh,
          gs_model_.optimParams.qtree_min_pixel_size,
          d_qtree_nodes_,
          rgb_img);
      qtree.subdivide();
      num_qtree_nodes_ = qtree.getNumLeaves();
      CUDA_CHECK(cudaMemcpy(d_num_qtree_nodes_, &num_qtree_nodes_, sizeof(size_t), cudaMemcpyHostToDevice));

      torch::Tensor image_tensor = torch::from_blob(reinterpret_cast<uint8_t*>(rgb_img.data<1>()),
                                                    {rgb_img.rows(), rgb_img.cols(), 3},
                                                    {rgb_img.cols() * 3, 3, 1},
                                                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
      cur_gt_img_                = image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;
      cur_gt_img_                = torch::clamp(cur_gt_img_, 0.f, 1.f);
      gt_img_list_.push_back(cur_gt_img_);
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::optimizeGS() {
      bool isKeyframe = false;

      // Update keyframe list
      if (num_valid_qtree_nodes_ > gs_model_.optimParams.kf_thresh) {
        isKeyframe = true;
      } else {
        // Only keep non-keyframes for ScanNet++ dataset
        if (!gs_model_.optimParams.keep_all_frames) {
          gs_cam_list_.pop_back();
          gt_img_list_.pop_back();
        }
      }

      if (num_valid_qtree_nodes_ != 0) {
        torch::NoGradGuard no_grad;
        gs_model_.Add_gaussians(d_positions_, d_colors_, d_scales_, num_valid_qtree_nodes_);
      }

      int iters = gs_model_.optimParams.kf_iters;
      if (!isKeyframe) {
        iters = gs_model_.optimParams.non_kf_iters;
      }

      std::vector<int> kf_indices = gs::get_random_indices(gt_img_list_.size());

      // Start online optimization
      for (int iter = 0; iter < iters; iter++) {
        auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam_, gs_model_);

        // Loss Computations
        auto loss = gs::l1_loss(image, cur_gt_img_);

        // Optimization
        loss.backward();
        gs_model_.optimizer->step();
        gs_model_.optimizer->zero_grad(true);

        // Store the cv::Mat rendered image for visualization
        if (iter == iters - 1) {
          auto rendered_img_tensor = image.detach().permute({1, 2, 0}).contiguous().to(torch::kCPU);
          rendered_img_tensor      = rendered_img_tensor.mul(255).clamp(0, 255).to(torch::kU8);
          auto cv_rendered_img     = cv::Mat(image.size(1), image.size(2), CV_8UC3, rendered_img_tensor.data_ptr());
          // cv::imshow("rendered", cv_rendered_img);
          // cv::waitKey(1);
        }
      }

      if (!isKeyframe) {
        int kf_iters = gs_model_.optimParams.random_kf_num;
        if (kf_indices.size() < kf_iters) {
          kf_iters = kf_indices.size();
        }
        for (int i = 0; i < kf_iters; i++) {
          auto kf_gt_img = gt_img_list_[kf_indices[i]];
          auto kf_gs_cam = gs_cam_list_[kf_indices[i]];

          auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(kf_gs_cam, gs_model_);
          auto loss                                                      = gs::l1_loss(image, kf_gt_img);
          loss.backward();
          gs_model_.optimizer->step();
          gs_model_.optimizer->zero_grad(true);
        }
      }
      torch::cuda::synchronize();
    }

    template <typename T>
    void
    GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::runGS(Camera& camera,
                                                                              VoxelContainer<T>& container,
                                                                              cupanutils::cugeoutils::CUDAMatrixuc3& rgb_img,
                                                                              cupanutils::cugeoutils::CUDAMatrixf& depth_img) {
      size_t free_byte;
      size_t total_byte;
      cudaMemGetInfo(&free_byte, &total_byte);
      if (free_byte < 100 * 1024 * 1024) {
        std::cout << "[GaussianContainer] Low GPU memory (" << free_byte / (1024 * 1024)
                  << " MB free). Skipping Gaussian Splatting update to avoid OOM." << std::endl;
        return;
      }
      setupGSCamera(camera);
      extractNodesQTree(camera, container, rgb_img, depth_img);
      checkNodes(camera, container, rgb_img, depth_img);
      optimizeGS();
    }

    template <typename T>
    void GaussianContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::optimizeGSFinal() {
      auto lambda = gs_model_.optimParams.lambda_dssim;
      auto iters  = gs_model_.optimParams.global_iters;

      for (int it = 0; it < iters; it++) {
        std::vector<int> indices = gs::get_random_indices(gt_img_list_.size());
        for (int i = 0; i < indices.size(); i++) {
          auto cur_gt_img = gt_img_list_[indices[i]];
          auto cur_gs_cam = gs_cam_list_[indices[i]];

          auto [image, viewspace_point_tensor, visibility_filter, radii] = gs::render(cur_gs_cam, gs_model_);

          // Loss Computations
          auto l1_loss   = gs::l1_loss(image, cur_gt_img);
          auto ssim_loss = gs::ssim(image, cur_gt_img, gs::conv_window, gs::window_size, gs::channel);
          auto loss      = (1.f - lambda) * l1_loss + lambda * (1.f - ssim_loss);

          // Optimization
          loss.backward();
          gs_model_.optimizer->step();
          gs_model_.optimizer->zero_grad(true);
        }
      }
      torch::cuda::synchronize();
    }

  } // namespace cugeoutils
} // namespace cupanutils
