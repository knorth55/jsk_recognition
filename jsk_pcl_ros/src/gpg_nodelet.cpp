// -*- mode: c++ -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2021, JSK Lab
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/o2r other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK Lab nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "jsk_pcl_ros/gpg.h"

#include <jsk_topic_tools/diagnostic_nodelet.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <sensor_msgs/PointCloud2.h>

namespace jsk_pcl_ros
{
  void GPG::onInit () {
    DiagnosticNodelet::onInit();
    pnh_->param("number_of_threads", num_of_threads_, 0);
    pnh_->param("number of k", k_, 30);
    pnh_->param("search radius", search_radius_, 0.0);
    pnh_->param("leaf size", leaf_size_, 0.01);
    onInitPostProcess();
  }

  void GPG::subscribe() {
      sub_ = pnh_->subscribe("input", 1, &GPG::processCloud, this);
  }

  void GPG::unsubscribe() {
      sub_.shutdown();
  }

  void GPG::processCloud(
    const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampled_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel.setSaveLeafLayout(true);
    voxel.setInputCloud(cloud);
    voxel.filter(*sampled_cloud);

    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> impl(num_of_threads_);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);
    impl.setInputCloud(sampled_cloud);
    impl.setSearchMethod (tree);
    impl.setKSearch(k_);
    impl.setRadiusSearch(search_radius_);
    impl.compute(*normal_cloud);
  }

}  // namespace jsk_pcl_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (jsk_pcl_ros::GPG, nodelet::Nodelet);
