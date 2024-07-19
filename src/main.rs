use std::vec;

use nalgebra::{self as na, ComplexField, Transform};

// get skew-symmetric matrix
fn skew(vector : na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ans = na::Matrix3::zeros();
    ans[(0, 1)] = -vector[2];
    ans[(0, 2)] = vector[1];
    ans[(1, 0)] = vector[3];
    ans[(1, 2)] = -vector[0];
    ans[(2, 0)] = -vector[1];
    ans[(2, 1)] = vector[0];
    
    ans
}

fn calculate_j(theta : f64, n :&na::Vector3<f64>) -> na::Matrix3<f64> {
    let i3 = na::Matrix3::<f64>::identity();

    // special judge
    if theta > 1e-6 {
       (theta.sin() / theta) * i3 + (1.0 - theta.cos()) / theta * skew(*n) 
       + (1.0 - theta.sin() / theta) * n * n.transpose()
    }else {
        return i3
    }
}

// from se(3) -> SE(3)
fn exp_map(v : &na::Vector6<f64>) -> na::Isometry3<f64> {
    let p_v = v.fixed_slice::<3, 1>(0, 0);
    let r_v = v.fixed_slice::<3, 1>(3, 0);

    let theta = r_v.norm();
    let n = r_v / theta;
    // through olinde rodrigues to get R
    let r = theta.cos() * na::Matrix3::<f64>::identity()  
            + (1.0 - theta.cos()) * n * n.transpose()
            + theta.sin() * skew(n);
    
    // through Jp get translation vector
    let j = calculate_j(theta, &n);

    // build S3(3)
    let t = na::Translation3::from(j * p_v);
    let r = na::Rotation3::from_matrix(&r);
    na::Isometry3::from_parts(t, r.into())

}

// from SE(3) -> se(3)
fn log_map(v : na::Isometry3::<f64>) -> na::Vector6<f64> {
    let t = v.translation.vector;
    let quat = v.rotation;

    // quaternion to angel-axis
    let theta = 2.0 * quat.scalar().acos();
    let n = quat.vector() / ((0.5 * theta).sin() + 1e-6);
    let r_v = theta * n;
    
    // t =  Jp --> p = t* J^-1
    let j = calculate_j(theta, &n);
    let j_r = j.try_inverse().unwrap();
    let p_v = j_r * t;

    let mut ret = na::Vector6::<f64>::zeros();
    ret.fixed_slice_mut::<3,1>(0, 0).copy_from(&p_v);
    ret.fixed_slice_mut::<3,1>(3, 0).copy_from(&r_v);

    return ret;


}

// Jacobian transformed point SE(3)
fn exp_map_jacobian(Transformed_point : &na::Point3<f64>) -> na::Matrix3x6<f64> {
    let mut ret = na::Matrix3x6::zeros();
    ret.fixed_slice_mut::<3, 3>(0, 0)
        .copy_from(&na::Matrix3::<f64>::identity());
    ret.fixed_slice_mut::<3,3>(0, 3)
        .copy_from(&(-skew(Transformed_point.coords)));

    return ret;
}

// projects a point in camera frame to images
fn project(
    // fx, fy, cx, cy
    params : &na::Vector4<f64>, pt : &na::Point3<f64>) -> na::Point2<f64> {
        na::Point2::<f64>::new(
            params[0] * pt.x / pt.z + params[2],
            params[1] * pt.y / pt.z + params[3],
        )
    }

    /// Jacobian of projection wrt 3D point in camera frame 
/// ref slam book eq. 6.43
fn proj_jacobian_wrt_point(
    // fx, fy, cx, cy
    camera_model: &na::Vector4<f64>,
    transformed_pt: &na::Point3<f64>,  
) -> na::Matrix2x3<f64> {
    na::Matrix2x3::<f64>::new(
        camera_model[0]/transformed_pt.z, 
        0.0, 
        -transformed_pt.x*camera_model[0]/transformed_pt.z.powi(2), 
        0.0,
        camera_model[1]/transformed_pt.z, 
        -transformed_pt.y*camera_model[1]/transformed_pt.z.powi(2), 
    )
}

struct Calibration<'a> {
    model_pts : &'a Vec<na::Point3<f64>>,
    image_pts_set : &'a Vec<Vec<na::Point2<f64>>>,
}

impl <'a> Calibration<'a> {
    
    // state: fx, fy ,cx, cy and all se(3) =  4 + + * num_images
    fn decode_param(
        &self,
        param : &na::DVector<f64>,
    ) -> (na::Vector4<f64>, Vec<na::Isometry3<f64>>) {
        let camera_model = param.fixed_slice::<4, 1>(0, 0).clone_owned();

        let transforms = self.
        image_pts_set
        .iter()
        .enumerate()
        .map(
            |(i, _)| 
            {
                let se3: na::Vector6<f64> = param.fixed_slice::<6,1>(4+6*i, 0).clone_owned(); 
                exp_map(&se3)
            }
        ).collect::<Vec<_>>(); 
                                
        (camera_model, transforms)
    }
}

fn main() {
    // create table 11 * 11 int the xy plane
    let mut source_pts : Vec<na::Point3<f64>> = Vec::new();
    for i in -5..6 {
        for j in -5..6 {
            source_pts.push(na::Point3::<f64>::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
        }
    }

     // Ground truth camera model
     let camera_model = na::Vector4::<f64>::new(540.0, 540.0, 320.0, 240.0); // fx, fy, cx, cy

     // Ground truth camera-from-model transforms for three "images"
     // new: translation, axisangle
     let transforms = vec![
         na::Isometry3::<f64>::new(
             na::Vector3::<f64>::new(-0.1, 0.1, 2.0),
             na::Vector3::<f64>::new(-0.2, 0.2, 0.2),
         ),
         na::Isometry3::<f64>::new(
             na::Vector3::<f64>::new(-0.1, -0.1, 2.0),
             na::Vector3::<f64>::new(0.2, -0.2, 0.2),
         ),
         na::Isometry3::<f64>::new(
             na::Vector3::<f64>::new(0.1, 0.1, 2.0),
             na::Vector3::<f64>::new(-0.2, -0.2, -0.2),
         ),
     ];

     // transform from model frame to camera frame
     let transform_pts = transforms
     .iter()
     .map(|t| source_pts.iter().map(|p| t * p).collect::<Vec<_>>()).collect::<Vec<_>>();
    
     // project to image
     
}
