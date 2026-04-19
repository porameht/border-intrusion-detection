use tracing::info;

use crate::types::{ChangeMask, GeoBbox, GeoTile, Tile};

/// Tile an image into inference-ready chips, only in changed regions (SAHI-style)
pub fn tile_changed_regions(
    image: &GeoTile,
    mask: &ChangeMask,
    tile_size: u32,
    overlap: f32,
) -> Vec<Tile> {
    let h = image.pixels.shape()[1] as u32;
    let w = image.pixels.shape()[2] as u32;
    let step = (tile_size as f32 * (1.0 - overlap)) as u32;

    let mut tiles = Vec::new();
    let mut grid_row = 0u32;

    let mut y = 0u32;
    while y + tile_size <= h {
        let mut grid_col = 0u32;
        let mut x = 0u32;
        while x + tile_size <= w {
            // Check if any pixel in this tile region is changed
            if region_has_change(mask, x, y, tile_size) {
                let tile = extract_tile(image, x, y, tile_size, grid_row, grid_col);
                tiles.push(tile);
            }
            x += step;
            grid_col += 1;
        }
        y += step;
        grid_row += 1;
    }

    let total_possible = ((h / step) * (w / step)) as usize;
    let pct_skipped = if total_possible > 0 {
        ((total_possible - tiles.len()) as f64 / total_possible as f64) * 100.0
    } else {
        0.0
    };

    info!(
        "Tiled: {} tiles extracted, {:.1}% skipped (no change)",
        tiles.len(),
        pct_skipped
    );

    tiles
}

/// Tile entire image without change mask (detect-everything mode)
pub fn tile_all(image: &GeoTile, tile_size: u32, overlap: f32) -> Vec<Tile> {
    let h = image.pixels.shape()[1] as u32;
    let w = image.pixels.shape()[2] as u32;
    let step = (tile_size as f32 * (1.0 - overlap)) as u32;

    let mut tiles = Vec::new();
    let mut grid_row = 0u32;

    let mut y = 0u32;
    while y + tile_size <= h {
        let mut grid_col = 0u32;
        let mut x = 0u32;
        while x + tile_size <= w {
            let tile = extract_tile(image, x, y, tile_size, grid_row, grid_col);
            tiles.push(tile);
            x += step;
            grid_col += 1;
        }
        y += step;
        grid_row += 1;
    }

    info!("Tiled: {} tiles (full coverage)", tiles.len());
    tiles
}

fn region_has_change(mask: &ChangeMask, x: u32, y: u32, size: u32) -> bool {
    let h = mask.mask.nrows() as u32;
    let w = mask.mask.ncols() as u32;

    for py in y..(y + size).min(h) {
        for px in x..(x + size).min(w) {
            if mask.mask[[py as usize, px as usize]] {
                return true;
            }
        }
    }
    false
}

fn extract_tile(image: &GeoTile, x: u32, y: u32, size: u32, grid_row: u32, grid_col: u32) -> Tile {
    let bands = image.pixels.shape()[0];
    let img_h = image.pixels.shape()[1] as u32;
    let img_w = image.pixels.shape()[2] as u32;
    let ts = size as usize;

    let mut pixels = ndarray::Array3::<f32>::zeros((bands, ts, ts));

    for b in 0..bands {
        for ty in 0..size {
            for tx in 0..size {
                let sy = (y + ty).min(img_h - 1) as usize;
                let sx = (x + tx).min(img_w - 1) as usize;
                pixels[[b, ty as usize, tx as usize]] = image.pixels[[b, sy, sx]];
            }
        }
    }

    // Interpolate geo bbox for this tile
    let bbox = interpolate_bbox(&image.bbox, img_w, img_h, x, y, size);

    Tile {
        pixels,
        grid_pos: (grid_row, grid_col),
        bbox,
    }
}

fn interpolate_bbox(parent: &GeoBbox, img_w: u32, img_h: u32, x: u32, y: u32, size: u32) -> GeoBbox {
    let lon_range = parent.max_lon - parent.min_lon;
    let lat_range = parent.max_lat - parent.min_lat;

    let px_lon = lon_range / img_w as f64;
    let px_lat = lat_range / img_h as f64;

    GeoBbox {
        min_lon: parent.min_lon + x as f64 * px_lon,
        min_lat: parent.min_lat + y as f64 * px_lat,
        max_lon: parent.min_lon + (x + size) as f64 * px_lon,
        max_lat: parent.min_lat + (y + size) as f64 * px_lat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn make_test_tile(h: usize, w: usize) -> GeoTile {
        GeoTile {
            pixels: Array3::<f32>::ones((3, h, w)),
            bbox: GeoBbox {
                min_lon: 100.0,
                min_lat: 15.0,
                max_lon: 101.0,
                max_lat: 16.0,
            },
            epsg: 4326,
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_tile_all() {
        let image = make_test_tile(640, 640);
        let tiles = tile_all(&image, 640, 0.0);
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn test_tile_with_overlap() {
        let image = make_test_tile(1280, 1280);
        let tiles = tile_all(&image, 640, 0.0);
        assert_eq!(tiles.len(), 4); // 2x2 grid
    }

    #[test]
    fn test_tile_changed_only() {
        let image = make_test_tile(1280, 1280);

        // Only top-left quadrant has change
        let mut mask_data = Array2::<bool>::default((1280, 1280));
        mask_data[[100, 100]] = true;

        let mask = ChangeMask {
            mask: mask_data,
            bbox: image.bbox,
            magnitude: Array2::<f32>::zeros((1280, 1280)),
        };

        let tiles = tile_changed_regions(&image, &mask, 640, 0.0);
        assert_eq!(tiles.len(), 1, "Only 1 tile should have change");
    }
}
