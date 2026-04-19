use anyhow::Result;
use std::path::Path;
use tracing::info;

use crate::types::ChangeAlert;

/// Write alerts as GeoJSON file (openable in QGIS)
pub fn write_geojson(alerts: &[ChangeAlert], output: &Path) -> Result<()> {
    let features: Vec<geojson::Feature> = alerts.iter().map(alert_to_feature).collect();

    let collection = geojson::FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let json = serde_json::to_string_pretty(&geojson::GeoJson::FeatureCollection(collection))?;
    std::fs::write(output, json)?;

    info!("Wrote {} alerts to {}", alerts.len(), output.display());
    Ok(())
}

fn alert_to_feature(alert: &ChangeAlert) -> geojson::Feature {
    let bbox = &alert.bbox;
    let geometry = geojson::Geometry::new(geojson::Value::Polygon(vec![vec![
        vec![bbox.min_lon, bbox.min_lat],
        vec![bbox.max_lon, bbox.min_lat],
        vec![bbox.max_lon, bbox.max_lat],
        vec![bbox.min_lon, bbox.max_lat],
        vec![bbox.min_lon, bbox.min_lat], // close ring
    ]]));

    let mut properties = serde_json::Map::new();
    properties.insert("id".to_string(), serde_json::json!(alert.id));
    properties.insert(
        "confidence".to_string(),
        serde_json::json!(alert.combined_confidence),
    );
    properties.insert(
        "priority".to_string(),
        serde_json::json!(format!("{:?}", alert.priority)),
    );
    properties.insert(
        "detection_count".to_string(),
        serde_json::json!(alert.detections.len()),
    );
    properties.insert(
        "timestamp".to_string(),
        serde_json::json!(alert.timestamp.to_rfc3339()),
    );

    let det_summaries: Vec<serde_json::Value> = alert
        .detections
        .iter()
        .map(|d| {
            serde_json::json!({
                "class": format!("{:?}", d.class),
                "confidence": d.confidence,
                "source": format!("{:?}", d.source),
                "lon": d.location.lon,
                "lat": d.location.lat,
            })
        })
        .collect();
    properties.insert("detections".to_string(), serde_json::json!(det_summaries));

    geojson::Feature {
        bbox: None,
        geometry: Some(geometry),
        id: Some(geojson::feature::Id::String(alert.id.clone())),
        properties: Some(properties),
        foreign_members: None,
    }
}

/// Write alerts as simple JSON
pub fn write_json(alerts: &[ChangeAlert], output: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(alerts)?;
    std::fs::write(output, json)?;
    info!("Wrote {} alerts to {}", alerts.len(), output.display());
    Ok(())
}

/// Print alert summary to stdout
pub fn print_summary(alerts: &[ChangeAlert]) {
    if alerts.is_empty() {
        println!("No alerts generated.");
        return;
    }

    println!("\n=== RIMRUA Alert Summary ===");
    println!("Total alerts: {}", alerts.len());

    use crate::types::AlertPriority::*;
    let (mut critical, mut high, mut medium, mut low) = (0, 0, 0, 0);
    for a in alerts {
        match a.priority {
            Critical => critical += 1,
            High => high += 1,
            Medium => medium += 1,
            Low => low += 1,
        }
    }

    println!("  CRITICAL: {critical}");
    println!("  HIGH:     {high}");
    println!("  MEDIUM:   {medium}");
    println!("  LOW:      {low}");

    println!("\nDetails:");
    for alert in alerts {
        println!(
            "  [{}] {:?} conf={:.2} detections={} @ ({:.4}, {:.4})",
            alert.id,
            alert.priority,
            alert.combined_confidence,
            alert.detections.len(),
            alert.bbox.min_lon,
            alert.bbox.min_lat,
        );
    }
    println!();
}
