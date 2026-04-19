mod alert;
mod change;
mod detect;
mod fuse;
mod ingest;
mod tile;
mod types;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "rimrua")]
#[command(about = "Border intrusion detection pipeline using satellite imagery")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run full pipeline from config file
    Pipeline { config: PathBuf },

    /// Run change detection between two images
    Change {
        before: PathBuf,
        after: PathBuf,
        /// "sar" or "optical"
        #[arg(long, default_value = "sar")]
        mode: String,
        #[arg(long, default_value = "0.9")]
        threshold: f32,
    },

    /// Run ML detection on an image
    Detect {
        image: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "0.5")]
        confidence: f32,
        #[arg(long, default_value = "640")]
        tile_size: u32,
    },

    /// Show pipeline info
    Info,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Pipeline { config } => run_pipeline(&config),
        Commands::Change { before, after, mode, threshold } => {
            run_change(&before, &after, &mode, threshold)
        }
        Commands::Detect { image, model, confidence, tile_size } => {
            run_detect(&image, &model, confidence, tile_size)
        }
        Commands::Info => { print_info(); Ok(()) }
    }
}

fn run_pipeline(config_path: &PathBuf) -> Result<()> {
    info!("Loading config: {}", config_path.display());
    let config_str = std::fs::read_to_string(config_path)?;
    let config: types::PipelineConfig = toml::from_str(&config_str)?;

    let mut results: Vec<types::Detection> = Vec::new();

    // SAR path
    if let (Some(before_path), Some(after_path)) =
        (&config.input.sar_before, &config.input.sar_after)
    {
        info!("=== SAR Change Detection ===");
        let before = ingest::read_geotiff(&PathBuf::from(before_path))?;
        let after = ingest::read_geotiff(&PathBuf::from(after_path))?;
        let mask = change::sar_coherence(&before, &after, 5, config.change.sar_coherence_threshold)?;
        let tiles = tile::tile_changed_regions(&after, &mask, config.detect.tile_size, config.detect.overlap);
        if !tiles.is_empty() {
            results.extend(detect::detect_tiles(&tiles, &config.detect.model, config.detect.confidence, 0.45, types::DataSource::Sar)?);
        }
    }

    // Optical path
    if let (Some(before_path), Some(after_path)) =
        (&config.input.optical_before, &config.input.optical_after)
    {
        info!("=== Optical Change Detection ===");
        let before = ingest::read_geotiff(&PathBuf::from(before_path))?;
        let after = ingest::read_geotiff(&PathBuf::from(after_path))?;
        let mask = change::ndvi_change(&before, &after, config.change.ndvi_threshold)?;
        let tiles = tile::tile_changed_regions(&after, &mask, config.detect.tile_size, config.detect.overlap);
        if !tiles.is_empty() {
            results.extend(detect::detect_tiles(&tiles, &config.detect.model, config.detect.confidence, 0.45, types::DataSource::Optical)?);
        }
    }

    // Fuse + Alert
    info!("=== Fusion ===");
    let fused = fuse::bayesian_fuse(&results, &fuse::SourceWeights::default());
    let alerts = fuse::to_alerts(fused);

    let output_path = PathBuf::from(&config.alert.output);
    match config.alert.format.as_str() {
        "json" => alert::write_json(&alerts, &output_path)?,
        _ => alert::write_geojson(&alerts, &output_path)?,
    }
    alert::print_summary(&alerts);
    Ok(())
}

fn run_change(before: &PathBuf, after: &PathBuf, mode: &str, threshold: f32) -> Result<()> {
    let before_tile = ingest::read_geotiff(before)?;
    let after_tile = ingest::read_geotiff(after)?;

    let mask = match mode {
        "optical" => change::ndvi_change(&before_tile, &after_tile, threshold)?,
        _ => change::sar_coherence(&before_tile, &after_tile, 5, threshold)?,
    };

    let changed: usize = mask.mask.iter().filter(|&&v| v).count();
    let total = mask.mask.len();
    println!("Change detected: {}/{} pixels ({:.1}%)", changed, total, (changed as f64 / total as f64) * 100.0);
    Ok(())
}

fn run_detect(image: &PathBuf, model: &PathBuf, confidence: f32, tile_size: u32) -> Result<()> {
    let img = ingest::read_geotiff(image)?;
    let tiles = tile::tile_all(&img, tile_size, 0.2);
    let detections = detect::detect_tiles(&tiles, &model.display().to_string(), confidence, 0.45, types::DataSource::Sar)?;

    println!("Detections: {}", detections.len());
    for det in &detections {
        println!("  {:?} conf={:.2} @ ({:.4}, {:.4})", det.class, det.confidence, det.location.lon, det.location.lat);
    }
    Ok(())
}

fn print_info() {
    println!("rimrua v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Border intrusion detection pipeline (SAR + Optical)");
    println!();
    println!("Pipeline: Ingest → Change Detection → Tile → Detect → Fuse → Alert");
    println!();
    println!("Modes:");
    println!("  SAR      Sentinel-1 coherence — works through clouds, day/night");
    println!("  Optical  Sentinel-2 NDVI — vegetation change, clear sky only");
    println!();
    println!("Usage:");
    println!("  rimrua pipeline config.toml");
    println!("  rimrua change before.tif after.tif --mode sar");
    println!("  rimrua change before.tif after.tif --mode optical");
    println!("  rimrua detect image.tif --model yolov8n.onnx");
}
