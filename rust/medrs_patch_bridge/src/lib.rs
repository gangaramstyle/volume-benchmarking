use ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn trilinear_sample(volume: &Array3<f32>, x: f32, y: f32, z: f32) -> f32 {
    let sx = volume.shape()[0] as isize;
    let sy = volume.shape()[1] as isize;
    let sz = volume.shape()[2] as isize;

    if x < 0.0 || y < 0.0 || z < 0.0 || x > (sx - 1) as f32 || y > (sy - 1) as f32 || z > (sz - 1) as f32 {
        return 0.0;
    }

    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let z0 = z.floor() as isize;
    let x1 = (x0 + 1).min(sx - 1);
    let y1 = (y0 + 1).min(sy - 1);
    let z1 = (z0 + 1).min(sz - 1);

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let dz = z - z0 as f32;

    let c000 = volume[[x0 as usize, y0 as usize, z0 as usize]];
    let c100 = volume[[x1 as usize, y0 as usize, z0 as usize]];
    let c010 = volume[[x0 as usize, y1 as usize, z0 as usize]];
    let c110 = volume[[x1 as usize, y1 as usize, z0 as usize]];
    let c001 = volume[[x0 as usize, y0 as usize, z1 as usize]];
    let c101 = volume[[x1 as usize, y0 as usize, z1 as usize]];
    let c011 = volume[[x0 as usize, y1 as usize, z1 as usize]];
    let c111 = volume[[x1 as usize, y1 as usize, z1 as usize]];

    let c00 = c000 * (1.0 - dx) + c100 * dx;
    let c01 = c001 * (1.0 - dx) + c101 * dx;
    let c10 = c010 * (1.0 - dx) + c110 * dx;
    let c11 = c011 * (1.0 - dx) + c111 * dx;
    let c0 = c00 * (1.0 - dy) + c10 * dy;
    let c1 = c01 * (1.0 - dy) + c11 * dy;
    c0 * (1.0 - dz) + c1 * dz
}

fn nearest_sample(volume: &Array3<f32>, x: f32, y: f32, z: f32) -> f32 {
    let sx = volume.shape()[0] as isize;
    let sy = volume.shape()[1] as isize;
    let sz = volume.shape()[2] as isize;

    let xi = x.round() as isize;
    let yi = y.round() as isize;
    let zi = z.round() as isize;

    if xi < 0 || yi < 0 || zi < 0 || xi >= sx || yi >= sy || zi >= sz {
        return 0.0;
    }
    volume[[xi as usize, yi as usize, zi as usize]]
}

fn world_to_voxel(affine_inv: &ndarray::ArrayView2<f32>, wx: f32, wy: f32, wz: f32) -> (f32, f32, f32) {
    let vx = affine_inv[[0, 0]] * wx + affine_inv[[0, 1]] * wy + affine_inv[[0, 2]] * wz + affine_inv[[0, 3]];
    let vy = affine_inv[[1, 0]] * wx + affine_inv[[1, 1]] * wy + affine_inv[[1, 2]] * wz + affine_inv[[1, 3]];
    let vz = affine_inv[[2, 0]] * wx + affine_inv[[2, 1]] * wy + affine_inv[[2, 2]] * wz + affine_inv[[2, 3]];
    (vx, vy, vz)
}

fn window_params(wc: f32, ww: f32) -> (f32, f32, f32) {
    let ww_safe = ww.max(1e-6);
    let wmin = wc - 0.5 * ww_safe;
    let wmax = wc + 0.5 * ww_safe;
    let inv = 2.0 / (wmax - wmin).max(1e-6);
    (wmin, wmax, inv)
}

fn apply_window(value: f32, wmin: f32, wmax: f32, inv: f32) -> f32 {
    let clipped = value.clamp(wmin, wmax);
    ((clipped - wmin) * inv) - 1.0
}

#[pyfunction]
fn sample_patches_trilinear<'py>(
    py: Python<'py>,
    volume_xyz: PyReadonlyArray3<'py, f32>,
    coords_xyz: PyReadonlyArray4<'py, f32>,
) -> PyResult<&'py PyArray3<f32>> {
    let volume = volume_xyz.as_array().to_owned();
    let coords = coords_xyz.as_array().to_owned();

    if coords.shape().len() != 4 || coords.shape()[3] != 3 {
        return Err(PyValueError::new_err("coords_xyz must have shape (P, H, W, 3)"));
    }

    let p = coords.shape()[0];
    let h = coords.shape()[1];
    let w = coords.shape()[2];
    let mut out = vec![0.0_f32; p * h * w];

    py.allow_threads(|| {
        out.par_chunks_mut(h * w)
            .enumerate()
            .for_each(|(patch_idx, out_patch)| {
                for y in 0..h {
                    for x in 0..w {
                        let cx = coords[[patch_idx, y, x, 0]];
                        let cy = coords[[patch_idx, y, x, 1]];
                        let cz = coords[[patch_idx, y, x, 2]];
                        out_patch[y * w + x] = trilinear_sample(&volume, cx, cy, cz);
                    }
                }
            });
    });

    let out_arr = Array3::from_shape_vec((p, h, w), out)
        .map_err(|_| PyValueError::new_err("failed to reshape trilinear output"))?;
    Ok(out_arr.into_pyarray(py))
}

#[pyfunction]
fn sample_asymmetric_patches_fused<'py>(
    py: Python<'py>,
    volume_xyz: PyReadonlyArray3<'py, f32>,
    affine_inv: PyReadonlyArray2<'py, f32>,
    centers_a_world: PyReadonlyArray2<'py, f32>,
    centers_b_world: PyReadonlyArray2<'py, f32>,
    rotation_matrix: PyReadonlyArray2<'py, f32>,
    window_a_wc: f32,
    window_a_ww: f32,
    window_b_wc: f32,
    window_b_ww: f32,
    a_native_no_interp: bool,
) -> PyResult<(&'py PyArray3<f32>, &'py PyArray3<f32>)> {
    let volume = volume_xyz.as_array().to_owned();
    let affine = affine_inv.as_array().to_owned();
    let centers_a = centers_a_world.as_array().to_owned();
    let centers_b = centers_b_world.as_array().to_owned();
    let rot = rotation_matrix.as_array().to_owned();

    if affine.shape() != [4, 4] {
        return Err(PyValueError::new_err("affine_inv must have shape (4, 4)"));
    }
    if centers_a.shape().len() != 2 || centers_a.shape()[1] != 3 {
        return Err(PyValueError::new_err("centers_a_world must have shape (N, 3)"));
    }
    if centers_b.shape().len() != 2 || centers_b.shape()[1] != 3 {
        return Err(PyValueError::new_err("centers_b_world must have shape (N, 3)"));
    }
    if centers_a.shape()[0] != centers_b.shape()[0] {
        return Err(PyValueError::new_err(
            "centers_a_world and centers_b_world must have the same N",
        ));
    }
    if rot.shape() != [3, 3] {
        return Err(PyValueError::new_err("rotation_matrix must have shape (3, 3)"));
    }

    let n = centers_a.shape()[0];
    let mut out_a = vec![0.0_f32; n * 16 * 16];
    let mut out_b = vec![0.0_f32; n * 16 * 16];

    let extent_x = 32.0_f32;
    let extent_y = 32.0_f32;
    let step_x = extent_x / 15.0_f32;
    let step_y = extent_y / 15.0_f32;
    let start_x = -0.5_f32 * extent_x;
    let start_y = -0.5_f32 * extent_y;
    let (a_wmin, a_wmax, a_inv) = window_params(window_a_wc, window_a_ww);
    let (b_wmin, b_wmax, b_inv) = window_params(window_b_wc, window_b_ww);

    py.allow_threads(|| {
        out_a
            .par_chunks_mut(16 * 16)
            .zip(out_b.par_chunks_mut(16 * 16))
            .enumerate()
            .for_each(|(patch_idx, (a_patch, b_patch))| {
                let ca_x = centers_a[[patch_idx, 0]];
                let ca_y = centers_a[[patch_idx, 1]];
                let ca_z = centers_a[[patch_idx, 2]];
                let cb_x = centers_b[[patch_idx, 0]];
                let cb_y = centers_b[[patch_idx, 1]];
                let cb_z = centers_b[[patch_idx, 2]];

                for yi in 0..16 {
                    let oy = start_y + yi as f32 * step_y;
                    for xi in 0..16 {
                        let ox = start_x + xi as f32 * step_x;
                        let out_idx = yi * 16 + xi;

                        // Anchor A: native axis-aligned path.
                        let wa_x = ca_x + ox;
                        let wa_y = ca_y + oy;
                        let wa_z = ca_z;
                        let (va_x, va_y, va_z) = world_to_voxel(&affine.view(), wa_x, wa_y, wa_z);
                        let a_raw = if a_native_no_interp {
                            nearest_sample(&volume, va_x, va_y, va_z)
                        } else {
                            trilinear_sample(&volume, va_x, va_y, va_z)
                        };
                        a_patch[out_idx] = apply_window(a_raw, a_wmin, a_wmax, a_inv);

                        // Anchor B: rotated + trilinear interpolation.
                        let rb_x = rot[[0, 0]] * ox + rot[[0, 1]] * oy;
                        let rb_y = rot[[1, 0]] * ox + rot[[1, 1]] * oy;
                        let rb_z = rot[[2, 0]] * ox + rot[[2, 1]] * oy;
                        let wb_x = cb_x + rb_x;
                        let wb_y = cb_y + rb_y;
                        let wb_z = cb_z + rb_z;
                        let (vb_x, vb_y, vb_z) = world_to_voxel(&affine.view(), wb_x, wb_y, wb_z);
                        let b_raw = trilinear_sample(&volume, vb_x, vb_y, vb_z);
                        b_patch[out_idx] = apply_window(b_raw, b_wmin, b_wmax, b_inv);
                    }
                }
            });
    });

    let out_a_arr = Array3::from_shape_vec((n, 16, 16), out_a)
        .map_err(|_| PyValueError::new_err("failed to reshape fused A output"))?;
    let out_b_arr = Array3::from_shape_vec((n, 16, 16), out_b)
        .map_err(|_| PyValueError::new_err("failed to reshape fused B output"))?;

    Ok((out_a_arr.into_pyarray(py), out_b_arr.into_pyarray(py)))
}

#[pymodule]
fn medrs_patch_bridge(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_patches_trilinear, m)?)?;
    m.add_function(wrap_pyfunction!(sample_asymmetric_patches_fused, m)?)?;
    Ok(())
}
