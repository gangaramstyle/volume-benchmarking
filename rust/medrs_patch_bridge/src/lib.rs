use ndarray::{Array3, Array4};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pyfunction]
fn sample_patches_trilinear<'py>(
    py: Python<'py>,
    volume_xyz: PyReadonlyArray3<'py, f32>,
    coords_xyz: PyReadonlyArray4<'py, f32>,
) -> PyResult<&'py PyArray3<f32>> {
    let volume = volume_xyz.as_array().to_owned();
    let coords = coords_xyz.as_array();

    if coords.shape().len() != 4 || coords.shape()[3] != 3 {
        return Err(PyValueError::new_err(
            "coords_xyz must have shape (P, H, W, 3)",
        ));
    }

    let p = coords.shape()[0];
    let h = coords.shape()[1];
    let w = coords.shape()[2];

    let mut out = Array3::<f32>::zeros((p, h, w));

    for patch_idx in 0..p {
        for y in 0..h {
            for x in 0..w {
                let cx = coords[[patch_idx, y, x, 0]];
                let cy = coords[[patch_idx, y, x, 1]];
                let cz = coords[[patch_idx, y, x, 2]];
                out[[patch_idx, y, x]] = trilinear_sample(&volume, cx, cy, cz);
            }
        }
    }

    Ok(out.into_pyarray(py))
}

#[pymodule]
fn medrs_patch_bridge(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_patches_trilinear, m)?)?;
    Ok(())
}
