//! fancam-core — High-performance automated fancam generation engine
//!
//! This crate provides the core video processing pipeline for generating
//! stabilized, vertical (9:16) cropped videos locked onto a target person.
//!
//! # Architecture
//!
//! The pipeline consists of several stages:
//! - **Detection**: YOLOv8-based person detection
//! - **Identification**: ArcFace-based face recognition
//! - **Tracking**: Kalman filter for smooth camera movement
//! - **Rendering**: SIMD-accelerated crop and resize
//! - **Discovery**: Multi-person identity clustering for group videos
//!
//! # Example
//!
//! ```rust,no_run
//! use fancam_core::pipeline::Pipeline;
//!
//! let pipeline = Pipeline::load(
//!     "yolov8n.onnx",
//!     "arcface.onnx",
//!     "reference_face.jpg",
//!     0.6, // similarity threshold
//! ).expect("Failed to load pipeline");
//!
//! let (mut analyzer, mut renderer) = pipeline.into_parts();
//! // Use analyzer and renderer in your processing loop...
//! ```

#![cfg_attr(not(test), warn(clippy::unwrap_used))]
#![cfg_attr(not(test), warn(clippy::expect_used))]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    rust_2018_idioms,
    unused_qualifications,
    missing_debug_implementations
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
// Deny unsafe code by default - explicit exceptions required
#![deny(unsafe_code)]

pub mod detection;
pub mod discovery;
pub mod error;
pub mod mode;
pub mod pipeline;
pub mod reid;
pub mod rendering;
pub mod runtime;
pub mod tracking;
pub mod video;

// Re-export error types
pub use error::{FancamError, PoisonExt, Result};

// Re-export anyhow for convenience in application code
pub use anyhow::{Error as AnyhowError, Result as AnyhowResult};
