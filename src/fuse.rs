use tracing::info;

use crate::types::{AlertPriority, ChangeAlert, Detection, GeoBbox, GeoPoint};

/// Bayesian fusion: combine detections from multiple sources
/// Higher weight = more trusted source
pub fn bayesian_fuse(
    detections: &[Detection],
    weights: &SourceWeights,
) -> Vec<FusedDetection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Cluster nearby detections using simple distance threshold
    let clusters = cluster_detections(detections, 0.001); // ~100m in degrees

    let mut fused = Vec::new();
    for cluster in &clusters {
        let combined = combine_cluster(cluster, weights);
        fused.push(combined);
    }

    info!(
        "Fusion: {} detections → {} clusters",
        detections.len(),
        fused.len()
    );

    fused
}

/// Source weights for Bayesian combination
pub struct SourceWeights {
    pub sar: f32,
    pub optical: f32,
}

impl Default for SourceWeights {
    fn default() -> Self {
        Self {
            sar: 0.4,
            optical: 0.35,
        }
    }
}

impl SourceWeights {
    pub fn for_source(&self, source: crate::types::DataSource) -> f32 {
        match source {
            crate::types::DataSource::Sar => self.sar,
            crate::types::DataSource::Optical => self.optical,
            crate::types::DataSource::Fused => (self.sar + self.optical) / 2.0,
        }
    }
}

pub struct FusedDetection {
    pub confidence: f32,
    pub priority: AlertPriority,
    pub detections: Vec<Detection>,
}

/// Simple DBSCAN-style clustering by geographic distance
fn cluster_detections(detections: &[Detection], eps: f64) -> Vec<Vec<&Detection>> {
    let mut visited = vec![false; detections.len()];
    let mut clusters: Vec<Vec<&Detection>> = Vec::new();

    for i in 0..detections.len() {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let mut cluster = vec![&detections[i]];

        // Find all neighbors within eps
        for j in (i + 1)..detections.len() {
            if visited[j] {
                continue;
            }
            let dist = geo_distance(&detections[i].location, &detections[j].location);
            if dist < eps {
                visited[j] = true;
                cluster.push(&detections[j]);
            }
        }

        clusters.push(cluster);
    }

    clusters
}

fn geo_distance(a: &GeoPoint, b: &GeoPoint) -> f64 {
    let dlat = a.lat - b.lat;
    let dlon = a.lon - b.lon;
    (dlat * dlat + dlon * dlon).sqrt()
}

fn combine_cluster(cluster: &[&Detection], weights: &SourceWeights) -> FusedDetection {
    // Bayesian confidence: P(event) = 1 - product(1 - p_i * w_i)
    let mut prob_no_event = 1.0f32;
    for det in cluster {
        let w = weights.for_source(det.source);
        prob_no_event *= 1.0 - (det.confidence * w);
    }
    let confidence = (1.0 - prob_no_event).clamp(0.0, 1.0);
    let priority = classify_priority(confidence, cluster.len());

    FusedDetection {
        confidence,
        priority,
        detections: cluster.iter().map(|d| (*d).clone()).collect(),
    }
}

fn classify_priority(confidence: f32, source_count: usize) -> AlertPriority {
    match (confidence, source_count) {
        (c, n) if c > 0.8 && n >= 2 => AlertPriority::Critical,
        (c, _) if c > 0.7 => AlertPriority::High,
        (c, _) if c > 0.4 => AlertPriority::Medium,
        _ => AlertPriority::Low,
    }
}

/// Convert fused detections into ChangeAlerts
pub fn to_alerts(fused: Vec<FusedDetection>) -> Vec<ChangeAlert> {
    fused
        .into_iter()
        .enumerate()
        .map(|(i, f)| {
            let bbox = compute_cluster_bbox(&f.detections);
            ChangeAlert {
                id: format!("alert-{:04}", i),
                detections: f.detections,
                combined_confidence: f.confidence,
                priority: f.priority,
                bbox,
                timestamp: chrono::Utc::now(),
            }
        })
        .collect()
}

fn compute_cluster_bbox(detections: &[Detection]) -> GeoBbox {
    let mut min_lon = f64::INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut max_lat = f64::NEG_INFINITY;

    for det in detections {
        min_lon = min_lon.min(det.location.lon);
        min_lat = min_lat.min(det.location.lat);
        max_lon = max_lon.max(det.location.lon);
        max_lat = max_lat.max(det.location.lat);
    }

    GeoBbox {
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DataSource, Obb, TargetClass};

    fn make_det(lon: f64, lat: f64, conf: f32, src: DataSource) -> Detection {
        Detection {
            obb: Obb { cx: 0.0, cy: 0.0, w: 10.0, h: 10.0, angle: 0.0 },
            class: TargetClass::Vehicle,
            confidence: conf,
            location: GeoPoint { lon, lat },
            source: src,
        }
    }

    #[test]
    fn test_cluster_nearby() {
        let dets = vec![
            make_det(100.0, 15.0, 0.8, DataSource::Sar),
            make_det(100.0001, 15.0001, 0.7, DataSource::Optical),
            make_det(105.0, 20.0, 0.6, DataSource::Sar), // far away
        ];

        let clusters = cluster_detections(&dets, 0.001);
        assert_eq!(clusters.len(), 2, "Should have 2 clusters");
        assert_eq!(clusters[0].len(), 2, "First cluster should have 2 detections");
        assert_eq!(clusters[1].len(), 1, "Second cluster should have 1 detection");
    }

    #[test]
    fn test_multi_source_higher_confidence() {
        let dets = vec![
            make_det(100.0, 15.0, 0.6, DataSource::Sar),
            make_det(100.0001, 15.0001, 0.6, DataSource::Optical),
        ];

        let fused = bayesian_fuse(&dets, &SourceWeights::default());
        assert_eq!(fused.len(), 1);
        let weights = SourceWeights::default();
        let single_sar = 1.0 - (1.0 - 0.6 * weights.sar);
        assert!(
            fused[0].confidence > single_sar,
            "Multi-source ({}) should exceed single source ({})",
            fused[0].confidence,
            single_sar
        );
    }
}
