use anyhow::{Context, Result};
use ndarray::Array3;
use std::path::Path;
use tracing::info;

use crate::types::{GeoBbox, GeoTile};

/// Read a GeoTIFF file and return a GeoTile
pub fn read_geotiff(path: &Path) -> Result<GeoTile> {
    info!("Reading GeoTIFF: {}", path.display());

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open: {}", path.display()))?;

    let (width, height, bands, flat) = decode_tiff(file)?;
    info!("Image size: {}x{}, {} bands", width, height, bands);

    let pixels = Array3::from_shape_vec((bands, height as usize, width as usize), flat)
        .context("Pixel data size mismatch")?;

    // TODO: parse GeoTIFF tags (ModelTiepointTag + ModelPixelScaleTag) for real bbox
    let bbox = GeoBbox {
        min_lon: 99.85,
        min_lat: 20.38,
        max_lon: 99.95,
        max_lat: 20.45,
    };

    Ok(GeoTile {
        pixels,
        bbox,
        epsg: 4326,
        timestamp: chrono::Utc::now(),
        source: path.display().to_string(),
    })
}

fn decode_tiff(file: std::fs::File) -> Result<(u32, u32, usize, Vec<f32>)> {
    use std::io::BufReader;
    use tiff::decoder::{Decoder, DecodingResult};

    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader).context("Failed to open TIFF")?;
    let (width, height) = decoder.dimensions().context("Failed to read dimensions")?;
    let pixels_per_band = (width * height) as usize;

    let mut all_bands: Vec<Vec<f32>> = Vec::new();

    loop {
        let result = decoder.read_image().context("Failed to read TIFF band")?;
        let band_data: Vec<f32> = match result {
            DecodingResult::F32(data) => data,
            DecodingResult::U8(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U16(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U32(data) => data.into_iter().map(|v| v as f32).collect(),
            DecodingResult::I16(data) => data.into_iter().map(|v| v as f32).collect(),
            _ => anyhow::bail!("Unsupported TIFF pixel type"),
        };

        if band_data.len() == pixels_per_band {
            all_bands.push(band_data);
        } else if band_data.len() % pixels_per_band == 0 {
            // De-interleave: [R1,G1,B1, R2,G2,B2, ...] → [RRR..., GGG..., BBB...]
            let num_bands = band_data.len() / pixels_per_band;
            for b in 0..num_bands {
                let band: Vec<f32> = (0..pixels_per_band)
                    .map(|p| band_data[p * num_bands + b])
                    .collect();
                all_bands.push(band);
            }
        } else {
            all_bands.push(band_data);
        }

        if decoder.more_images() {
            decoder.next_image().context("Failed to advance to next TIFF IFD")?;
        } else {
            break;
        }
    }

    // Band-sequential flat buffer for Array3::from_shape_vec
    let bands = all_bands.len();
    let capacity = bands * pixels_per_band;
    let mut flat = Vec::with_capacity(capacity);
    for band_data in all_bands {
        flat.extend(band_data);
    }

    Ok((width, height, bands, flat))
}
