use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex;
use tracing::info;

use crate::types::{ChangeMask, GeoTile};

// ─── Optical: NDVI Change Detection ───────────────────────────────────────

/// NDVI = (NIR - Red) / (NIR + Red)
/// Detects vegetation removal: camp clearing, new roads
pub fn ndvi_change(before: &GeoTile, after: &GeoTile, threshold: f32) -> Result<ChangeMask> {
    info!("Computing NDVI change, threshold={}", threshold);

    let (bands, h, w) = before.dims();
    let nir_band = if bands >= 4 { 3 } else { 2 };

    let mut mask = Array2::<bool>::default((h, w));
    let mut magnitude = Array2::<f32>::zeros((h, w));
    let mut changed = 0usize;

    // Single pass: compute NDVI inline, no intermediate arrays
    for y in 0..h {
        for x in 0..w {
            let ndvi_b = ndvi_pixel(&before.pixels, y, x, nir_band);
            let ndvi_a = ndvi_pixel(&after.pixels, y, x, nir_band);
            let diff = (ndvi_a - ndvi_b).abs();
            magnitude[[y, x]] = diff;
            if diff > threshold {
                mask[[y, x]] = true;
                changed += 1;
            }
        }
    }

    let pct = (changed as f64 / (h * w) as f64) * 100.0;
    info!("NDVI change: {}/{} pixels ({:.1}%)", changed, h * w, pct);

    Ok(ChangeMask { mask, bbox: before.bbox, magnitude })
}

fn ndvi_pixel(pixels: &ndarray::Array3<f32>, y: usize, x: usize, nir_band: usize) -> f32 {
    let nir = pixels[[nir_band, y, x]];
    let red = pixels[[0, y, x]];
    let sum = nir + red;
    if sum > 1e-6 { (nir - red) / sum } else { 0.0 }
}

// ─── SAR: Coherence Change Detection ──────────────────────────────────────

/// Coherence = |<s1 * conj(s2)>| / sqrt(<|s1|^2> * <|s2|^2>)
/// Low coherence = ground disturbed (footprints, tire tracks, digging)
/// Works through clouds and at night
pub fn sar_coherence(
    before: &GeoTile,
    after: &GeoTile,
    window_size: usize,
    threshold: f32,
) -> Result<ChangeMask> {
    info!("Computing SAR coherence, window={}, threshold={}", window_size, threshold);

    let (_, h, w) = before.dims();

    let s1 = to_complex(&before.pixels);
    let s2 = to_complex(&after.pixels);

    // Integral images for O(1) windowed sums
    let (cross_re_sat, cross_im_sat) = cross_integral(&s1, &s2, h, w);
    let power1_sat = power_integral(&s1, h, w);
    let power2_sat = power_integral(&s2, h, w);

    let half = window_size / 2;
    let mut mask = Array2::<bool>::default((h, w));
    let mut magnitude = Array2::<f32>::zeros((h, w));
    let mut changed = 0usize;

    for y in half..(h.saturating_sub(half)) {
        for x in half..(w.saturating_sub(half)) {
            let y0 = y.saturating_sub(half);
            let x0 = x.saturating_sub(half);
            let y1 = (y + half).min(h - 1);
            let x1 = (x + half).min(w - 1);

            let cross_re = rect_sum(&cross_re_sat, y0, x0, y1, x1);
            let cross_im = rect_sum(&cross_im_sat, y0, x0, y1, x1);
            let p1 = rect_sum(&power1_sat, y0, x0, y1, x1);
            let p2 = rect_sum(&power2_sat, y0, x0, y1, x1);

            let cross_norm = (cross_re * cross_re + cross_im * cross_im).sqrt();
            let denom = (p1 * p2).sqrt();
            let coh = if denom > 1e-10 { (cross_norm / denom) as f32 } else { 1.0 };

            magnitude[[y, x]] = 1.0 - coh;
            if coh < threshold {
                mask[[y, x]] = true;
                changed += 1;
            }
        }
    }

    let pct = (changed as f64 / (h * w) as f64) * 100.0;
    info!("SAR change: {}/{} pixels ({:.1}%)", changed, h * w, pct);

    Ok(ChangeMask { mask, bbox: before.bbox, magnitude })
}

fn to_complex(pixels: &ndarray::Array3<f32>) -> Vec<Complex<f64>> {
    let bands = pixels.shape()[0];
    let h = pixels.shape()[1];
    let w = pixels.shape()[2];
    let mut complex = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let re = pixels[[0, y, x]] as f64;
            let im = if bands >= 2 { pixels[[1, y, x]] as f64 } else { 0.0 };
            complex.push(Complex::new(re, im));
        }
    }
    complex
}

// Summed area table (integral image) for O(1) rectangle queries
type Sat = Vec<Vec<f64>>;

fn cross_integral(s1: &[Complex<f64>], s2: &[Complex<f64>], h: usize, w: usize) -> (Sat, Sat) {
    let mut re = vec![vec![0.0f64; w + 1]; h + 1];
    let mut im = vec![vec![0.0f64; w + 1]; h + 1];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let cross = s1[idx] * s2[idx].conj();
            re[y + 1][x + 1] = cross.re + re[y][x + 1] + re[y + 1][x] - re[y][x];
            im[y + 1][x + 1] = cross.im + im[y][x + 1] + im[y + 1][x] - im[y][x];
        }
    }
    (re, im)
}

fn power_integral(s: &[Complex<f64>], h: usize, w: usize) -> Sat {
    let mut sat = vec![vec![0.0f64; w + 1]; h + 1];
    for y in 0..h {
        for x in 0..w {
            let p = s[y * w + x].norm_sqr();
            sat[y + 1][x + 1] = p + sat[y][x + 1] + sat[y + 1][x] - sat[y][x];
        }
    }
    sat
}

fn rect_sum(sat: &Sat, y0: usize, x0: usize, y1: usize, x1: usize) -> f64 {
    sat[y1 + 1][x1 + 1] - sat[y0][x1 + 1] - sat[y1 + 1][x0] + sat[y0][x0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GeoBbox;
    use ndarray::Array3;

    fn make_tile(pixels: Array3<f32>) -> GeoTile {
        GeoTile {
            pixels,
            bbox: GeoBbox { min_lon: 0.0, min_lat: 0.0, max_lon: 1.0, max_lat: 1.0 },
            epsg: 4326,
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_ndvi_no_change() {
        let tile = make_tile(Array3::<f32>::ones((4, 10, 10)));
        let mask = ndvi_change(&tile, &tile, 0.1).unwrap();
        assert!(mask.mask.iter().all(|&v| !v));
    }

    #[test]
    fn test_ndvi_with_change() {
        let before = make_tile(Array3::from_shape_fn((4, 10, 10), |(b, _, _)| {
            if b == 3 { 0.8 } else { 0.2 }
        }));
        let after = make_tile(Array3::from_shape_fn((4, 10, 10), |(b, _, _)| {
            if b == 3 { 0.2 } else { 0.8 }
        }));
        let mask = ndvi_change(&before, &after, 0.1).unwrap();
        assert!(mask.mask.iter().all(|&v| v));
    }

    #[test]
    fn test_sar_identical_no_change() {
        let tile = make_tile(Array3::from_shape_fn((2, 20, 20), |(b, y, x)| {
            ((y * 20 + x + b) as f32).sin()
        }));
        let mask = sar_coherence(&tile, &tile, 5, 0.3).unwrap();
        assert_eq!(mask.mask.iter().filter(|&&v| v).count(), 0);
    }

    #[test]
    fn test_sar_different_has_change() {
        let before = make_tile(Array3::from_shape_fn((1, 20, 20), |(_, y, x)| {
            ((y * 20 + x) as f32).sin() * 100.0
        }));
        let after = make_tile(Array3::from_shape_fn((1, 20, 20), |(_, y, x)| {
            ((y * 20 + x) as f32).cos() * 100.0
        }));
        let mask = sar_coherence(&before, &after, 5, 0.9).unwrap();
        assert!(mask.mask.iter().filter(|&&v| v).count() > 0);
    }
}
