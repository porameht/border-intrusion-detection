use anyhow::{Context, Result};
use ndarray::Array3;
use ort::session::Session;
use tracing::info;

use crate::types::{DataSource, Detection, GeoBbox, GeoPoint, Obb, TargetClass, Tile};

/// Run YOLOv8 ONNX inference on a batch of tiles
pub fn detect_tiles(
    tiles: &[Tile],
    model_path: &str,
    confidence_threshold: f32,
    nms_threshold: f32,
    source: DataSource,
) -> Result<Vec<Detection>> {
    info!("Running detection on {} tiles, model={}", tiles.len(), model_path);

    let mut session = load_model(model_path)?;
    let mut all_detections = Vec::new();

    for tile in tiles {
        let raw = run_inference(&mut session, &tile.pixels)?;
        let dets = parse_yolo_output(&raw, confidence_threshold, &tile.bbox, source);
        all_detections.extend(dets);
    }

    let before_nms = all_detections.len();
    all_detections = geo_nms(&all_detections, nms_threshold);

    info!("Detections: {} raw → {} after NMS", before_nms, all_detections.len());
    Ok(all_detections)
}

fn load_model(path: &str) -> Result<Session> {
    info!("Loading ONNX model: {}", path);
    Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load model {path}: {e}"))
}

const MODEL_INPUT_SIZE: usize = 640;

fn run_inference(session: &mut Session, pixels: &Array3<f32>) -> Result<Vec<f32>> {
    let (c, h, w) = (pixels.shape()[0], pixels.shape()[1], pixels.shape()[2]);
    let ms = MODEL_INPUT_SIZE;

    // Normalize to 0-1 and resize to model input size in a single pass
    let max_val = pixels.iter().cloned().fold(0.0f32, f32::max);
    let scale = if max_val > 1.0 { 255.0f32.max(max_val) } else { 1.0 };

    let flat: Vec<f32> = if h != ms || w != ms {
        let sy = h as f32 / ms as f32;
        let sx = w as f32 / ms as f32;
        let mut buf = Vec::with_capacity(c * ms * ms);
        for band in 0..c {
            for y in 0..ms {
                for x in 0..ms {
                    let y0 = (y as f32 * sy).min((h - 1) as f32) as usize;
                    let x0 = (x as f32 * sx).min((w - 1) as f32) as usize;
                    buf.push(pixels[[band, y0, x0]] / scale);
                }
            }
        }
        buf
    } else {
        pixels.iter().map(|&v| v / scale).collect()
    };

    let shape = vec![1usize, c, ms, ms];

    let input_value =
        ort::value::Tensor::from_array((shape, flat)).map_err(|e| anyhow::anyhow!("{e}"))?;

    let outputs = session
        .run(ort::inputs![input_value])
        .map_err(|e| anyhow::anyhow!("Inference error: {e}"))?;

    let output_key = outputs.keys().next().context("No output tensor")?;
    let (_shape, data) = outputs[output_key]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(data.to_vec())
}

// ─── YOLOv8 Output Parsing ────────────────────────────────────────────────

/// Parse YOLOv8 output: [1, num_attrs, 8400]
/// OBB model: num_attrs = 4 (box) + 1 (angle) + num_classes
/// Detect model: num_attrs = 4 (box) + num_classes
fn parse_yolo_output(
    raw: &[f32],
    conf_threshold: f32,
    tile_bbox: &GeoBbox,
    source: DataSource,
) -> Vec<Detection> {
    let num_anchors = 8400usize;
    let tile_size = 640u32;

    let total = raw.len();
    let num_rows = total / num_anchors;
    if num_rows == 0 || total % num_anchors != 0 {
        return Vec::new();
    }

    // OBB format: [1, 20, 8400] = 4 (cx,cy,w,h) + 15 (class scores, already sigmoided) + 1 (angle)
    // Detect format: [1, 84, 8400] = 4 (cx,cy,w,h) + 80 (class logits, need sigmoid)
    let (is_obb, num_classes) = match num_rows {
        20 => (true, 15usize),
        84 => (false, 80usize),
        n => (false, n.saturating_sub(4)),
    };

    let mut detections = Vec::new();

    for i in 0..num_anchors {
        // Class scores start at row 4
        let mut best_class = 0usize;
        let mut best_score = 0.0f32;
        for c in 0..num_classes {
            let raw_val = raw[(4 + c) * num_anchors + i];
            // OBB scores are already sigmoided, detect scores need sigmoid
            let score = if is_obb { raw_val } else { sigmoid(raw_val) };
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }

        if best_score < conf_threshold {
            continue;
        }

        let cx = raw[i];
        let cy = raw[1 * num_anchors + i];
        let w = raw[2 * num_anchors + i];
        let h = raw[3 * num_anchors + i];
        // OBB: angle is the LAST row (row 19), not row 4
        let angle = if is_obb {
            raw[(num_rows - 1) * num_anchors + i]
        } else {
            0.0
        };

        let class = if is_obb {
            dota_to_target_class(best_class)
        } else {
            coco_to_target_class(best_class)
        };
        let location = pixel_to_geo(cx, cy, tile_bbox, tile_size);

        detections.push(Detection {
            obb: Obb { cx, cy, w, h, angle },
            class,
            confidence: best_score,
            location,
            source,
        });
    }

    detections
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Probabilistic IoU NMS (matches Ultralytics OBB NMS) ──────────────────

/// Cross-tile NMS using Probabilistic IoU for rotated boxes
/// Based on: https://arxiv.org/pdf/2106.06072v1.pdf
fn geo_nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<_> = detections.to_vec();
    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(sorted[i].clone());

        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            let iou = probiou(&sorted[i].obb, &sorted[j].obb);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Probabilistic IoU between two oriented bounding boxes
/// Treats OBBs as 2D Gaussian distributions and computes Bhattacharyya distance
fn probiou(a: &Obb, b: &Obb) -> f32 {
    let eps = 1e-7f32;

    // Covariance matrix components for box a: [[a1, c1], [c1, b1]]
    let (a1, b1, c1) = obb_covariance(a);
    let (a2, b2, c2) = obb_covariance(b);

    let dx = b.cx - a.cx;
    let dy = b.cy - a.cy;

    let sum_a = a1 + a2;
    let sum_b = b1 + b2;
    let sum_c = c1 + c2;

    let det_sum = sum_a * sum_b - sum_c * sum_c + eps;

    // Bhattacharyya distance terms
    let t1 = ((sum_a * dy * dy + sum_b * dx * dx) / (det_sum + eps)) * 0.25;
    let t2 = ((sum_c * (dx) * (-dy)) / (det_sum + eps)) * 0.5;

    let det_a = (a1 * b1 - c1 * c1).max(0.0);
    let det_b = (a2 * b2 - c2 * c2).max(0.0);
    let t3 = ((det_sum) / (4.0 * (det_a * det_b).sqrt() + eps) + eps).ln() * 0.5;

    let bd = (t1 + t2 + t3).clamp(eps, 100.0);
    let hd = (1.0 - (-bd).exp() + eps).sqrt();

    1.0 - hd // probiou = 1 - hellinger distance
}

/// Compute covariance matrix from OBB (Gaussian bounding box representation)
/// Returns (a, b, c) where cov = [[a, c], [c, b]]
fn obb_covariance(obb: &Obb) -> (f32, f32, f32) {
    let w_var = obb.w * obb.w / 12.0;
    let h_var = obb.h * obb.h / 12.0;
    let cos = obb.angle.cos();
    let sin = obb.angle.sin();
    let cos2 = cos * cos;
    let sin2 = sin * sin;

    let a = w_var * cos2 + h_var * sin2;
    let b = w_var * sin2 + h_var * cos2;
    let c = (w_var - h_var) * cos * sin;

    (a, b, c)
}

// ─── Class Mapping ────────────────────────────────────────────────────────

fn dota_to_target_class(class_id: usize) -> TargetClass {
    match class_id {
        0 => TargetClass::Aircraft,    // plane
        1 => TargetClass::Vessel,      // ship
        2 => TargetClass::Structure,   // storage tank
        3 | 4 | 5 | 6 | 13 | 14 => TargetClass::Structure,
        7 => TargetClass::Structure,   // harbor
        8 => TargetClass::Structure,   // bridge
        9 => TargetClass::Vehicle,     // large vehicle
        10 => TargetClass::Vehicle,    // small vehicle
        11 => TargetClass::Aircraft,   // helicopter
        12 => TargetClass::Structure,  // roundabout
        _ => TargetClass::Unknown,
    }
}

fn coco_to_target_class(class_id: usize) -> TargetClass {
    match class_id {
        2 => TargetClass::Vehicle,  // car
        5 => TargetClass::Vehicle,  // bus
        7 => TargetClass::Vehicle,  // truck
        4 => TargetClass::Aircraft, // airplane
        8 => TargetClass::Vessel,   // boat
        _ => TargetClass::Unknown,
    }
}

// ─── Utils ────────────────────────────────────────────────────────────────

pub fn pixel_to_geo(px: f32, py: f32, tile_bbox: &GeoBbox, tile_size: u32) -> GeoPoint {
    let lon = tile_bbox.min_lon
        + (px as f64 / tile_size as f64) * (tile_bbox.max_lon - tile_bbox.min_lon);
    let lat = tile_bbox.min_lat
        + (py as f64 / tile_size as f64) * (tile_bbox.max_lat - tile_bbox.min_lat);
    GeoPoint { lon, lat }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSource, TargetClass};

    #[test]
    fn test_probiou_identical() {
        let obb = Obb { cx: 50.0, cy: 50.0, w: 20.0, h: 10.0, angle: 0.3 };
        let iou = probiou(&obb, &obb);
        assert!(iou > 0.95, "probiou of identical OBBs should be ~1.0, got {}", iou);
    }

    #[test]
    fn test_probiou_no_overlap() {
        let a = Obb { cx: 0.0, cy: 0.0, w: 10.0, h: 10.0, angle: 0.0 };
        let b = Obb { cx: 1000.0, cy: 1000.0, w: 10.0, h: 10.0, angle: 0.0 };
        let iou = probiou(&a, &b);
        assert!(iou < 0.01, "probiou of far-apart OBBs should be ~0, got {}", iou);
    }

    #[test]
    fn test_probiou_rotated() {
        let a = Obb { cx: 50.0, cy: 50.0, w: 40.0, h: 10.0, angle: 0.0 };
        let b = Obb { cx: 50.0, cy: 50.0, w: 40.0, h: 10.0, angle: 1.5 }; // ~90 degrees
        let iou = probiou(&a, &b);
        // Same center but perpendicular → low overlap
        assert!(iou < 0.5, "Perpendicular OBBs should have low probiou, got {}", iou);
    }

    #[test]
    fn test_nms_suppression() {
        let dets = vec![
            Detection {
                obb: Obb { cx: 50.0, cy: 50.0, w: 20.0, h: 10.0, angle: 0.0 },
                class: TargetClass::Vehicle,
                confidence: 0.9,
                location: GeoPoint { lon: 100.0, lat: 15.0 },
                source: DataSource::Sar,
            },
            Detection {
                obb: Obb { cx: 52.0, cy: 51.0, w: 20.0, h: 10.0, angle: 0.0 },
                class: TargetClass::Vehicle,
                confidence: 0.7,
                location: GeoPoint { lon: 100.0, lat: 15.0 },
                source: DataSource::Sar,
            },
        ];
        let result = geo_nms(&dets, 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
