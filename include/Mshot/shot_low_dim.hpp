/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
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
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 *
 *
 */

#ifndef PCL_FEATURES_IMPL_SHOTLowDim_H_
#define PCL_FEATURES_IMPL_SHOTLowDim_H_

#include </home/sai/pcl_testing/3DLRF/include/Mshot/shot_low_dim.h>
#include </home/sai/pcl_testing/3DLRF/include/Mshot/shot_low_dim_lrf.h>
#include <utility>

// Useful constants.
#define PST_PI 3.1415926535897932384626433832795
#define PST_RAD_45 0.78539816339744830961566084581988
#define PST_RAD_90 1.5707963267948966192313216916398
#define PST_RAD_135 2.3561944901923449288469825374596
#define PST_RAD_180 PST_PI
#define PST_RAD_360 6.283185307179586476925286766558
#define PST_RAD_PI_7_8 2.7488935718910690836548129603691

const double zeroDoubleEps15 = 1E-15;
const float zeroFloatEps8 = 1E-8f;

//////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Check if val1 and val2 are equals.
  *
  * \param[in] val1 first number to check.
  * \param[in] val2 second number to check.
  * \param[in] zeroDoubleEps epsilon
  * \return true if val1 is equal to val2, false otherwise.
  */
inline bool
areEquals (double val1, double val2, double zeroDoubleEps = zeroDoubleEps15)
{
  return (fabs (val1 - val2)<zeroDoubleEps);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Check if val1 and val2 are equals.
  *
  * \param[in] val1 first number to check.
  * \param[in] val2 second number to check.
  * \param[in] zeroFloatEps epsilon
  * \return true if val1 is equal to val2, false otherwise.
  */
inline bool
areEquals (float val1, float val2, float zeroFloatEps = zeroFloatEps8)
{
  return (fabs (val1 - val2)<zeroFloatEps);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> bool
pcl::SHOTLowDimEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::initCompute ()
{
  if (!FeatureFromNormals<PointInT, PointNT, PointOutT>::initCompute ())
  {
    PCL_ERROR ("[pcl::%s::initCompute] Init failed.\n", getClassName ().c_str ());
    return (false);
  }

  // SHOT cannot work with k-search
  if (this->getKSearch () != 0)
  {
    PCL_ERROR(
      "[pcl::%s::initCompute] Error! Search method set to k-neighborhood. Call setKSearch(0) and setRadiusSearch( radius ) to use this class.\n",
      getClassName().c_str ());
    return (false);
  }

  // Default LRF estimation alg: SHOTLocalReferenceFrameEstimation
  typename SHOTLocalReferenceFrameEstimation<PointInT, PointRFT>::Ptr lrf_estimator(new SHOTLocalReferenceFrameEstimation<PointInT, PointRFT>());
  lrf_estimator->setRadiusSearch ((lrf_radius_ > 0 ? lrf_radius_ : search_radius_));
  lrf_estimator->setInputCloud (input_);
  lrf_estimator->setIndices (indices_);
  if (!fake_surface_)
    lrf_estimator->setSearchSurface(surface_);

  if (!FeatureWithLocalReferenceFrames<PointInT, PointRFT>::initLocalReferenceFrames (indices_->size (), lrf_estimator))
  {
    PCL_ERROR ("[pcl::%s::initCompute] Init failed.\n", getClassName ().c_str ());
    return (false);
  }

  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> void
pcl::SHOTLowDimEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::createBinDistanceShape (
    int index,
    const std::vector<int> &indices,
    std::vector<double> &bin_distance_shape)
{
  bin_distance_shape.resize (indices.size ());

  const PointRFT& current_frame = frames_->points[index];
  //if (!pcl_isfinite (current_frame.rf[0]) || !pcl_isfinite (current_frame.rf[4]) || !pcl_isfinite (current_frame.rf[11]))
    //return;

  Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

  unsigned nan_counter = 0;
  for (size_t i_idx = 0; i_idx < indices.size (); ++i_idx)
  {
    // check NaN normal
    const Eigen::Vector4f& normal_vec = normals_->points[indices[i_idx]].getNormalVector4fMap ();
    if (!pcl_isfinite (normal_vec[0]) ||
        !pcl_isfinite (normal_vec[1]) ||
        !pcl_isfinite (normal_vec[2]))
    {
      bin_distance_shape[i_idx] = std::numeric_limits<double>::quiet_NaN ();
      ++nan_counter;
    } else
    {
      //double cosineDesc = feat[i].rf[6]*normal[0] + feat[i].rf[7]*normal[1] + feat[i].rf[8]*normal[2];
      double cosineDesc = normal_vec.dot (current_frame_z);

      if (cosineDesc > 1.0)
        cosineDesc = 1.0;
      if (cosineDesc < - 1.0)
        cosineDesc = - 1.0;

      bin_distance_shape[i_idx] = ((1.0 + cosineDesc) * nr_shape_bins_) / 2;
    }
  }
  if (nan_counter > 0)
    PCL_WARN ("[pcl::%s::createBinDistanceShape] Point %d has %d (%f%%) NaN normals in its neighbourhood\n",
      getClassName ().c_str (), index, nan_counter, (static_cast<float>(nan_counter)*100.f/static_cast<float>(indices.size ())));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> void
pcl::SHOTLowDimEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::normalizeHistogram (
    Eigen::VectorXf &shot, int desc_length)
{
	// Normalization is performed by considering the L2 norm
	// and not the sum of bins, as reported in the ECCV paper.
	// This is due to additional experiments performed by the authors after its pubblication,
	// where L2 normalization turned out better at handling point density variations.
  double acc_norm = 0;
  for (int j = 0; j < desc_length; j++)
    acc_norm += shot[j] * shot[j];
  acc_norm = sqrt (acc_norm);
  for (int j = 0; j < desc_length; j++)
    shot[j] /= static_cast<float> (acc_norm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> void
pcl::SHOTLowDimEstimationBase<PointInT, PointNT, PointOutT, PointRFT>::interpolateSingleChannel (
    const std::vector<int> &indices,
    const std::vector<float> &sqr_dists,
    const int index,
    std::vector<double> &binDistance,
    const int nr_bins,
    Eigen::VectorXf &shot)
{
  const Eigen::Vector4f& central_point = (*input_)[(*indices_)[index]].getVector4fMap ();
  const PointRFT& current_frame = (*frames_)[index];

  Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
  Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
  Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

  for (size_t i_idx = 0; i_idx < indices.size (); ++i_idx)
  {
    if (!pcl_isfinite(binDistance[i_idx]))
      continue;

    Eigen::Vector4f delta = surface_->points[indices[i_idx]].getVector4fMap () - central_point;
    delta[3] = 0;

    // Compute the Euclidean norm
   double distance = sqrt (sqr_dists[i_idx]);

    if (areEquals (distance, 0.0))
      continue;

    //cout << "delta: "<< delta<< " ";

    double xInFeatRef = delta.dot (current_frame_x);
    double yInFeatRef = delta.dot (current_frame_y);
    double zInFeatRef = delta.dot (current_frame_z);

    //cout << "InFeatRef: " << xInFeatRef << " " << yInFeatRef << " "<< zInFeatRef<<endl;

    // To avoid numerical problems afterwards
    if (fabs (yInFeatRef) < 1E-30)
      yInFeatRef  = 0;
    if (fabs (xInFeatRef) < 1E-30)
      xInFeatRef  = 0;
    if (fabs (zInFeatRef) < 1E-30)
      zInFeatRef  = 0;


    unsigned char bit4 = ((yInFeatRef > 0) || ((yInFeatRef == 0.0) && (xInFeatRef < 0))) ? 1 : 0;
    unsigned char bit3 = static_cast<unsigned char> (((xInFeatRef > 0) || ((xInFeatRef == 0.0) && (yInFeatRef > 0))) ? !bit4 : bit4);

    //cout << "bit4 "<<bit4<<" "<<"bit3 "<<bit3;

    assert (bit3 == 0 || bit3 == 1);


    int desc_index = (bit4<<3) + (bit3<<2);

    //cout << "desc_index "<< desc_index<< endl;

    desc_index = desc_index << 1;

    //cout << "desc_index "<< desc_index << endl;


    if ((xInFeatRef * yInFeatRef > 0) || (xInFeatRef == 0.0))
      desc_index += (fabs (xInFeatRef) >= fabs (yInFeatRef)) ? 0 : 4;
    else
      desc_index += (fabs (xInFeatRef) > fabs (yInFeatRef)) ? 4 : 0;

    desc_index += zInFeatRef > 0 ? 1 : 0;

    // 2 RADII--- commented this la...see what happens
    desc_index += (distance > radius1_2_) ? 2 : 0;

    //int step_index = static_cast<int>(floor (binDistance[i_idx] +0.5));
    int volume_index = desc_index * (nr_bins/*+1*/);

    //Interpolation on the cosine (adjacent bins in the histogram)
    //binDistance[i_idx] -= step_index;
    //double intWeight = (1- fabs (binDistance[i_idx]));

    if (binDistance[i_idx] > 0)
      shot[volume_index /*+ ((step_index+1) % nr_bins)*/] += static_cast<float> (binDistance[i_idx]);
    else
      shot[volume_index /*+ ((step_index - 1 + nr_bins) % nr_bins)*/] += - static_cast<float> (binDistance[i_idx]);
/*
    //Interpolation on the distance (adjacent husks)

    if (distance > radius1_2_)   //external sphere
    {
      double radiusDistance = (distance - radius3_4_) / radius1_2_;

      if (distance > radius3_4_) //most external sector, votes only for itself
        intWeight += 1 - radiusDistance;  //peso=1-d
      else  //3/4 of radius, votes also for the internal sphere
      {
        intWeight += 1 + radiusDistance;
        shot[(desc_index - 2) * (nr_bins+1) + step_index] -= static_cast<float> (radiusDistance);
      }
    }
    else    //internal sphere
    {
      double radiusDistance = (distance - radius1_4_) / radius1_2_;

      if (distance < radius1_4_) //most internal sector, votes only for itself
        intWeight += 1 + radiusDistance;  //weight=1-d
      else  //3/4 of radius, votes also for the external sphere
      {
        intWeight += 1 - radiusDistance;
        shot[(desc_index + 2) * (nr_bins+1) + step_index] += static_cast<float> (radiusDistance);
      }
    }
*/
    /// INTERpolation
    /*
    //Interpolation on the inclination (adjacent vertical volumes)
    double inclinationCos = zInFeatRef / distance;
    if (inclinationCos < - 1.0)
      inclinationCos = - 1.0;
    if (inclinationCos > 1.0)
      inclinationCos = 1.0;

    double inclination = acos (inclinationCos);

    assert (inclination >= 0.0 && inclination <= PST_RAD_180);

    if (inclination > PST_RAD_90 || (fabs (inclination - PST_RAD_90) < 1e-30 && zInFeatRef <= 0))
    {
      double inclinationDistance = (inclination - PST_RAD_135) / PST_RAD_90;
      if (inclination > PST_RAD_135)
        intWeight += 1 - inclinationDistance;
      else
      {
        intWeight += 1 + inclinationDistance;
        assert ((desc_index + 1) * (nr_bins+1) + step_index >= 0 && (desc_index + 1) * (nr_bins+1) + step_index < descLength_);
        shot[(desc_index + 1) * (nr_bins+1) + step_index] -= static_cast<float> (inclinationDistance);
      }
    }
    else
    {
      double inclinationDistance = (inclination - PST_RAD_45) / PST_RAD_90;
      if (inclination < PST_RAD_45)
        intWeight += 1 + inclinationDistance;
      else
      {
        intWeight += 1 - inclinationDistance;
        assert ((desc_index - 1) * (nr_bins+1) + step_index >= 0 && (desc_index - 1) * (nr_bins+1) + step_index < descLength_);
        shot[(desc_index - 1) * (nr_bins+1) + step_index] += static_cast<float> (inclinationDistance);
      }
    }

    if (yInFeatRef != 0.0 || xInFeatRef != 0.0)
    {
      //Interpolation on the azimuth (adjacent horizontal volumes)
      double azimuth = atan2 (yInFeatRef, xInFeatRef);

      int sel = desc_index >> 2;
      double angularSectorSpan = PST_RAD_45;
      double angularSectorStart = - PST_RAD_PI_7_8;

      double azimuthDistance = (azimuth - (angularSectorStart + angularSectorSpan*sel)) / angularSectorSpan;

      assert ((azimuthDistance < 0.5 || areEquals (azimuthDistance, 0.5)) && (azimuthDistance > - 0.5 || areEquals (azimuthDistance, - 0.5)));

      azimuthDistance = (std::max)(- 0.5, std::min (azimuthDistance, 0.5));

      if (azimuthDistance > 0)
      {
        intWeight += 1 - azimuthDistance;
        int interp_index = (desc_index + 4) % maxAngularSectors_;
        assert (interp_index * (nr_bins+1) + step_index >= 0 && interp_index * (nr_bins+1) + step_index < descLength_);
        shot[interp_index * (nr_bins+1) + step_index] += static_cast<float> (azimuthDistance);
      }
      else
      {
        int interp_index = (desc_index - 4 + maxAngularSectors_) % maxAngularSectors_;
        assert (interp_index * (nr_bins+1) + step_index >= 0 && interp_index * (nr_bins+1) + step_index < descLength_);
        intWeight += 1 + azimuthDistance;
        shot[interp_index * (nr_bins+1) + step_index] -= static_cast<float> (azimuthDistance);
      }

    }
    */

    assert (volume_index /*+ step_index*/ >= 0 &&  volume_index /*+ step_index*/ < descLength_);
    //shot[volume_index /*+ step_index*/] += static_cast<float> (intWeight);
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> void
pcl::SHOTLowDimEstimation<PointInT, PointNT, PointOutT, PointRFT>::computePointSHOT (
  const int index, const std::vector<int> &indices, const std::vector<float> &sqr_dists, Eigen::VectorXf &shot)
{
  //Skip the current feature if the number of its neighbors is not sufficient for its description
  if (indices.size () < 5)
  {
    PCL_WARN ("[pcl::%s::computePointSHOT] Warning! Neighborhood has less than 5 vertexes. Aborting description of point with index %d\n",
                  getClassName ().c_str (), (*indices_)[index]);

    shot.setConstant(descLength_, 1, std::numeric_limits<float>::quiet_NaN () );

    return;
  }

   // Clear the resultant shot
  std::vector<double> binDistanceShape;
  this->createBinDistanceShape (index, indices, binDistanceShape);

  // Interpolate
  shot.setZero ();
  interpolateSingleChannel (indices, sqr_dists, index, binDistanceShape, nr_shape_bins_, shot);

  // Normalize the final histogram
  this->normalizeHistogram (shot, descLength_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, typename PointRFT> void
pcl::SHOTLowDimEstimation<PointInT, PointNT, PointOutT, PointRFT>::computeFeature (pcl::PointCloud<PointOutT> &output)
{
    //nr_grid_sector_ = 2;// added this
    //nr_shape_bins_ = 1;// added this

  descLength_ = nr_grid_sector_ * (nr_shape_bins_+1);// chnaged from (nr_shape_bins_+1) to 1

  sqradius_ = search_radius_ * search_radius_;
  radius3_4_ = (search_radius_*3) / 4;
  radius1_4_ = search_radius_ / 4;
  radius1_2_ = search_radius_ / 2;

  //assert(descLength_ == 352);

  shot_.setZero (descLength_);

  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  std::vector<int> nn_indices (k_);
  std::vector<float> nn_dists (k_);

  output.is_dense = true;
  // Iterating over the entire index vector
  for (size_t idx = 0; idx < indices_->size (); ++idx)
  {
    bool lrf_is_nan = false;
    const PointRFT& current_frame = (*frames_)[idx];
    if (!pcl_isfinite (current_frame.x_axis[0]) ||
        !pcl_isfinite (current_frame.y_axis[0]) ||
        !pcl_isfinite (current_frame.z_axis[0]))
    {
      PCL_WARN ("[pcl::%s::computeFeature] The local reference frame is not valid! Aborting description of point with index %d\n",
        getClassName ().c_str (), (*indices_)[idx]);
      lrf_is_nan = true;
    }

    if (!isFinite ((*input_)[(*indices_)[idx]]) ||
        lrf_is_nan ||
        this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
    {
      // Copy into the resultant cloud
      for (int d = 0; d < descLength_; ++d)
        output.points[idx].descriptor[d] = std::numeric_limits<float>::quiet_NaN ();
      for (int d = 0; d < 9; ++d)
        output.points[idx].rf[d] = std::numeric_limits<float>::quiet_NaN ();

      output.is_dense = false;
      continue;
    }

    // Estimate the SHOT descriptor at each patch
    computePointSHOT (static_cast<int> (idx), nn_indices, nn_dists, shot_);

    // Copy into the resultant cloud
    for (int d = 0; d < descLength_; ++d)
      output.points[idx].descriptor[d] = shot_[d];
    for (int d = 0; d < 3; ++d)
    {
      output.points[idx].rf[d + 0] = frames_->points[idx].x_axis[d];
      output.points[idx].rf[d + 3] = frames_->points[idx].y_axis[d];
      output.points[idx].rf[d + 6] = frames_->points[idx].z_axis[d];
    }
  }
}



#define PCL_INSTANTIATE_SHOTEstimationBase(T,NT,OutT,RFT) template class PCL_EXPORTS pcl::SHOTEstimationBase<T,NT,OutT,RFT>;
#define PCL_INSTANTIATE_SHOTEstimation(T,NT,OutT,RFT) template class PCL_EXPORTS pcl::SHOTEstimation<T,NT,OutT,RFT>;

#endif    // PCL_FEATURES_IMPL_SHOT_H_
