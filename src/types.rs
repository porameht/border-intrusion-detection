use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A geo-referenced raster tile ready for processing
pub struct GeoTile {
    /// Pixel data [bands, height, width]
    pub pixels: ndarray::Array3<f32>,
    /// Geographic bounding box
    pub bbox: GeoBbox,
    /// Coordinate reference system EPSG code
    pub epsg: u32,
    /// Acquisition timestamp
    pub timestamp: DateTime<Utc>,
    /// Source file path
    pub source: String,
}

/// Geographic bounding box in lat/lon (EPSG:4326)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoBbox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

/// A point in geographic coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lon: f64,
    pub lat: f64,
}

/// Binary change mask from change detection
pub struct ChangeMask {
    /// Boolean mask [height, width] — true = changed
    pub mask: ndarray::Array2<bool>,
    /// Geographic bounding box
    pub bbox: GeoBbox,
    /// Change magnitude per pixel (0.0 - 1.0)
    pub magnitude: ndarray::Array2<f32>,
}

/// An inference-ready image tile
pub struct Tile {
    /// Pixel data normalized for model input [channels, height, width]
    pub pixels: ndarray::Array3<f32>,
    /// Which row/col in the tiling grid
    pub grid_pos: (u32, u32),
    /// Geographic bounding box of this tile
    pub bbox: GeoBbox,
}

/// Oriented bounding box (rotated rectangle)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Obb {
    /// Center x in pixel coordinates
    pub cx: f32,
    /// Center y in pixel coordinates
    pub cy: f32,
    /// Width
    pub w: f32,
    /// Height
    pub h: f32,
    /// Rotation angle in radians
    pub angle: f32,
}

impl GeoTile {
    pub fn dims(&self) -> (usize, usize, usize) {
        let s = self.pixels.shape();
        (s[0], s[1], s[2])
    }
}

/// Target class from detection model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetClass {
    Vehicle,
    Structure,
    Aircraft,
    Vessel,
    Unknown,
}

/// A single detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Oriented bounding box in pixel coords
    pub obb: Obb,
    /// Target classification
    pub class: TargetClass,
    /// Confidence score 0.0 - 1.0
    pub confidence: f32,
    /// Geographic location (center of detection)
    pub location: GeoPoint,
    /// Source data type
    pub source: DataSource,
}

/// Where the detection came from
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataSource {
    Sar,
    Optical,
    Fused,
}

/// Alert priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// A fused alert combining multiple detections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeAlert {
    pub id: String,
    pub detections: Vec<Detection>,
    pub combined_confidence: f32,
    pub priority: AlertPriority,
    pub bbox: GeoBbox,
    pub timestamp: DateTime<Utc>,
}

/// Pipeline configuration loaded from TOML
#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    pub input: InputConfig,
    pub change: ChangeConfig,
    pub detect: DetectConfig,
    pub alert: AlertConfig,
}

#[derive(Debug, Deserialize)]
pub struct InputConfig {
    pub sar_before: Option<String>,
    pub sar_after: Option<String>,
    pub optical_before: Option<String>,
    pub optical_after: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChangeConfig {
    pub sar_coherence_threshold: f32,
    pub ndvi_threshold: f32,
}

#[derive(Debug, Deserialize)]
pub struct DetectConfig {
    pub model: String,
    pub confidence: f32,
    pub tile_size: u32,
    pub overlap: f32,
}

#[derive(Debug, Deserialize)]
pub struct AlertConfig {
    pub format: String,
    pub output: String,
}
