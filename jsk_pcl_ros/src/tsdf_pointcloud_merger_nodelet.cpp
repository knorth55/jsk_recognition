// -*- mode: c++ -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, JSK Lab
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

#define BOOST_PARAMETER_MAX_ARITY 7

#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <jsk_recognition_utils/pcl_ros_util.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include "jsk_pcl_ros/tsdf_pointcloud_merger.h"

namespace enc = sensor_msgs::image_encodings;

namespace jsk_pcl_ros
{
  void
  TSDFPointcloudMerger::onInit()
  {
    ConnectionBasedNodelet::onInit();
    always_subscribe_ = true;  // for mapping

    pnh_->param("device", device_, 0);
    pnh_->param("integrate_color", integrate_color_, false);
    pnh_->param("volume_size", volume_size_, pcl::device::kinfuLS::VOLUME_SIZE);
    pnh_->param("shifting_distance", shifting_distance_, pcl::device::kinfuLS::DISTANCE_THRESHOLD);
    pnh_->param<std::string>("frame_id", frame_id_, "base_footprint");
    pnh_->param("queue_size", queue_size_, 10);

    tf_listener_.reset(new tf::TransformListener());

    pub_cloud_ = advertise<sensor_msgs::PointCloud2>(*pnh_, "output/cloud", 1);
    pub_depth_ = advertise<sensor_msgs::Image>(*pnh_, "output/depth", 1);

    onInitPostProcess();
  }

  void
  TSDFPointcloudMerger::initTSDFPointcloudMerger(const sensor_msgs::CameraInfo::ConstPtr& caminfo_msg)
  {
    pcl::gpu::setDevice(device_);
    pcl::gpu::printShortCudaDeviceInfo(device_);

    const Eigen::Vector3f volume_size = Eigen::Vector3f::Constant (volume_size_);
    const Eigen::Vector3i volume_resolution
      (pcl::device::kinfuLS::VOLUME_X, pcl::device::kinfuLS::VOLUME_Y, pcl::device::kinfuLS::VOLUME_Z);
    const float default_tranc_dist = 0.03f;

    tsdf_volume_ = pcl::gpu::kinfuLS::TsdfVolume::Ptr(
      new pcl::gpu::kinfuLS::TsdfVolume(volume_resolution) );
    tsdf_volume_->setSize (volume_size);
    tsdf_volume_->setTsdfTruncDist (default_tranc_dist);
    tsdf_volume_->reset ();

    cyclical_.setDistanceThreshold (shifting_distance_);
    cyclical_.setVolumeSize (volume_size_, volume_size_, volume_size_);
    cyclical_.resetBuffer(tsdf_volume_);

    if (integrate_color_)
    {
      const int max_color_integration_weight = 2;
      color_volume_ = pcl::gpu::kinfuLS::ColorVolume::Ptr(
        new pcl::gpu::kinfuLS::ColorVolume(*tsdf_volume_, max_color_integration_weight));
      color_volume_->reset ();
    }
    depthRawScaled_.create (caminfo_msg->height, caminfo_msg->width);
    cyclical_.initBuffer(tsdf_volume_);
  }

  void
  TSDFPointcloudMerger::subscribe()
  {
    sub_camera_info_.subscribe(*pnh_, "input/camera_info", 1);
    sub_depth_.subscribe(*pnh_, "input/depth", 1);
    if (integrate_color_)
    {
      sub_color_.subscribe(*pnh_, "input/color", 1);
      sync_with_color_.reset(new message_filters::Synchronizer<SyncPolicyWithColor>(queue_size_));
      sync_with_color_->connectInput(sub_camera_info_, sub_depth_, sub_color_);
      sync_with_color_->registerCallback(boost::bind(&TSDFPointcloudMerger::update, this, _1, _2, _3));
    }
    else
    {
      sync_.reset(new message_filters::Synchronizer<SyncPolicy>(queue_size_));
      sync_->connectInput(sub_camera_info_, sub_depth_);
      sync_->registerCallback(boost::bind(&TSDFPointcloudMerger::update, this, _1, _2));
    }
  }

  void
  TSDFPointcloudMerger::unsubscribe()
  {
  }

  void
  TSDFPointcloudMerger::update(const sensor_msgs::CameraInfo::ConstPtr& caminfo_msg,
                const sensor_msgs::Image::ConstPtr& depth_msg)
  {
    update(caminfo_msg, depth_msg, sensor_msgs::ImageConstPtr());
  }

  void
  TSDFPointcloudMerger::update(const sensor_msgs::CameraInfo::ConstPtr& caminfo_msg,
                const sensor_msgs::Image::ConstPtr& depth_msg,
                const sensor_msgs::Image::ConstPtr& color_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);

    if ((depth_msg->height != caminfo_msg->height) || (depth_msg->width != caminfo_msg->width))
    {
      ROS_ERROR("Image size of input depth and camera info must be same. Depth: (%d, %d), Camera Info: (%d, %d)",
                depth_msg->height, depth_msg->width, caminfo_msg->height, caminfo_msg->width);
      return;
    }
    if (integrate_color_ && ((color_msg->height != caminfo_msg->height) || (color_msg->width != color_msg->width)))
    {
      ROS_ERROR("Image size of input color image and camera info must be same. Color: (%d, %d), Camera Info: (%d, %d)",
                color_msg->height, color_msg->width, caminfo_msg->height, caminfo_msg->width);
      return;
    }

    if (!tsdf_volume_ || !color_volume_)
    {
      initTSDFPointcloudMerger(caminfo_msg);
    }

    // depth: 32fc1 -> 16uc1
    cv::Mat depth;
    if (depth_msg->encoding == enc::TYPE_32FC1)
    {
      cv::Mat depth_32fc1 = cv_bridge::toCvShare(depth_msg, enc::TYPE_32FC1)->image;
      depth_32fc1 *= 1000.;
      depth_32fc1.convertTo(depth, CV_16UC1);
    }
    else if (depth_msg->encoding == enc::TYPE_16UC1)
    {
      depth = cv_bridge::toCvShare(depth_msg, enc::TYPE_16UC1)->image;
    }
    else
    {
      NODELET_FATAL("Unsupported depth image encoding: %s", depth_msg->encoding.c_str());
      return;
    }

    tf::StampedTransform tf_stamped;
    Eigen::Affine3d tmp_transform ;
    Eigen::Affine3f current_transform ;
    try {
      tf_listener_->waitForTransform(caminfo_msg->header.frame_id, frame_id_, ros::Time(0), ros::Duration(1.0));
      tf_listener_->lookupTransform(caminfo_msg->header.frame_id, frame_id_, ros::Time(0), tf_stamped);
      tf::transformTFToEigen(tf_stamped, tmp_transform);
      current_transform = tmp_transform.cast<float> ();
      Eigen::Matrix<float, 3, 3, Eigen::RowMajor> current_R = current_transform.rotation();
      Eigen::Vector3f current_t = current_transform.translation();

      // depth: cpu -> gpu
      depth_device_.upload(&(depth.data[0]), depth.cols * 2, depth.rows, depth.cols);

      pcl::device::kinfuLS::Intr intr(
        caminfo_msg->K[0], caminfo_msg->K[4], caminfo_msg->K[2], caminfo_msg->K[5]);
      const float3 device_volume_size = device_cast<const float3> (tsdf_volume_->getSize());
      const pcl::device::kinfuLS::Mat33  device_current_R = device_cast<const pcl::device::kinfuLS::Mat33> (current_R);
      const float3 device_current_t = device_cast<const float3> (current_t);

      pcl::device::kinfuLS::integrateTsdfVolume(
        depth_device_, intr, device_volume_size, device_current_R, device_current_t,
        tsdf_volume_->getTsdfTruncDist(), tsdf_volume_->data(), cyclical_.getBuffer(), depthRawScaled_);

      if (integrate_color_)
      {
        // color: cpu -> gpu
        if (color_msg->encoding == enc::BGR8)
        {
          cv_bridge::CvImagePtr tmp_image_ptr_ = cv_bridge::toCvCopy(color_msg, enc::RGB8);
          colors_device_.upload(&(tmp_image_ptr_->toImageMsg()->data[0]),
                                color_msg->step, color_msg->height, color_msg->width);
        }
        else
        {
          colors_device_.upload(&(color_msg->data[0]), color_msg->step, color_msg->height, color_msg->width);
        }

        pcl::device::kinfuLS::MapArr device_vmap;
        pcl::device::kinfuLS::createVMap(intr, depth_device_, device_vmap);
        pcl::device::kinfuLS::updateColorVolume(
          intr, tsdf_volume_->getTsdfTruncDist(), device_current_R, device_current_t, device_vmap,
          colors_device_, device_volume_size, color_volume_->data(), color_volume_->getMaxWeight());
      }

      // publish depth image
      if (pub_depth_.getNumSubscribers() > 0)
      {
        pcl::device::kinfuLS::DepthMap depth_gpu;
        // Eigen::Affine3f camera_pose = kinfu_->getCameraPose();
        if (!raycaster_)
        {
          raycaster_ = pcl::gpu::kinfuLS::RayCaster::Ptr(
            new pcl::gpu::kinfuLS::RayCaster(
              caminfo_msg->height, caminfo_msg->width,
              caminfo_msg->K[0], caminfo_msg->K[4],
              caminfo_msg->K[2], caminfo_msg->K[5]));
        }
        raycaster_->generateDepthImage(depth_gpu);

        int cols;
        std::vector<unsigned short> data;
        depth_gpu.download(data, cols);

        sensor_msgs::Image output_depth_msg;
        sensor_msgs::fillImage(output_depth_msg,
                               enc::TYPE_16UC1,
                               depth_gpu.rows(),
                               depth_gpu.cols(),
                               depth_gpu.cols() * 2,
                               reinterpret_cast<unsigned short*>(&data[0]));
        output_depth_msg.header = caminfo_msg->header;
        pub_depth_.publish(output_depth_msg);
      }

      // publish cloud
      if (pub_cloud_.getNumSubscribers() > 0)
      {
        pcl::gpu::DeviceArray<pcl::PointXYZ> cloud_buffer_device;
        pcl::gpu::DeviceArray<pcl::PointXYZ> extracted = tsdf_volume_->fetchCloud(cloud_buffer_device);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
        extracted.download(cloud_xyz->points);
        cloud_xyz->width = static_cast<int>(cloud_xyz->points.size());
        cloud_xyz->height = 1;

        sensor_msgs::PointCloud2 output_cloud_msg;
        if (integrate_color_)
        {
          pcl::gpu::DeviceArray<pcl::RGB> point_colors_device;
          color_volume_->fetchColors(extracted, point_colors_device);

          pcl::PointCloud<pcl::RGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::RGB>());
          point_colors_device.download(cloud_rgb->points);
          cloud_rgb->width = static_cast<int>(cloud_rgb->points.size());
          cloud_rgb->height = 1;

          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
          cloud->points.resize(cloud_xyz->points.size());

          for (size_t i = 0; i < cloud_xyz->points.size(); i++)
          {
            cloud->points[i].x = cloud_xyz->points[i].x;
            cloud->points[i].y = cloud_xyz->points[i].y;
            cloud->points[i].z = cloud_xyz->points[i].z;
            cloud->points[i].r = cloud_rgb->points[i].r;
            cloud->points[i].g = cloud_rgb->points[i].g;
            cloud->points[i].b = cloud_rgb->points[i].b;
          }
          cloud->width = cloud_xyz->width;
          cloud->height = cloud_xyz->height;

          pcl::toROSMsg(*cloud, output_cloud_msg);
        }
        else
        {
          pcl::toROSMsg(*cloud_xyz, output_cloud_msg);
        }
        output_cloud_msg.header.frame_id = frame_id_;
        pub_cloud_.publish(output_cloud_msg);
      }
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
    }
  }
}  // namespace jsk_pcl_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(jsk_pcl_ros::TSDFPointcloudMerger, nodelet::Nodelet);
