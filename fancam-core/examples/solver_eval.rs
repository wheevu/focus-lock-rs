use std::time::Instant;

use fancam_core::detection::BBox;
use fancam_core::observation::IdentityObservation;
use fancam_core::solver;
use fancam_core::tracklet::Tracklet;

const TRACKLET_COUNT: usize = 160;
const FRAMES_PER_TRACKLET: u64 = 8;

fn main() {
    let fixture = fixture_tracklets();
    let expected_tracklets = fixture.len();

    let started = Instant::now();
    let solved = solver::solve(&fixture);
    let elapsed = started.elapsed();

    let selected = solved.selected_identity_id;
    let selected_assignments = selected.map_or(0, |identity_id| {
        solved
            .assignments
            .iter()
            .filter(|assignment| assignment.identity_id == identity_id)
            .count()
    });
    let identity_coverage = selected_assignments as f64 / expected_tracklets as f64;
    let jitter_px = mean_camera_step_px(&solved.camera_path.keyframes);
    let throughput = expected_tracklets as f64 / elapsed.as_secs_f64().max(f64::EPSILON);

    println!("focus-lock deterministic solver/camera evaluation");
    println!("fixture_tracklets={expected_tracklets}");
    println!("assignments={}", solved.assignments.len());
    println!("selected_identity_id={selected:?}");
    println!("selected_identity_coverage={identity_coverage:.4}");
    println!("camera_keyframes={}", solved.camera_path.keyframes.len());
    println!("mean_camera_step_px={jitter_px:.4}");
    println!("solver_elapsed_ms={:.4}", elapsed.as_secs_f64() * 1000.0);
    println!("tracklets_per_second={throughput:.2}");

    assert_eq!(solved.assignments.len(), expected_tracklets);
    assert!(identity_coverage >= 0.95);
    assert!(jitter_px.is_finite());
    assert!(jitter_px <= 20.0);
}

fn fixture_tracklets() -> Vec<Tracklet> {
    (0..TRACKLET_COUNT)
        .map(|tracklet_id| {
            let mut tracklet = Tracklet::new(tracklet_id);
            let base_frame = tracklet_id as u64 * FRAMES_PER_TRACKLET;
            for offset in 0..FRAMES_PER_TRACKLET {
                let frame_index = base_frame + offset;
                let cx = 240.0 + frame_index as f32 * 1.35 + ((frame_index % 5) as f32 - 2.0) * 0.7;
                let cy = 420.0 + (frame_index as f32 / 18.0).sin() * 14.0;
                let bbox = bbox_from_center(cx, cy, 72.0, 168.0);
                let similarity = 0.82 - (tracklet_id % 4) as f32 * 0.01;
                let impostor = 0.24 + (tracklet_id % 3) as f32 * 0.01;
                let margin = similarity - impostor;
                tracklet.push(
                    frame_index,
                    bbox,
                    IdentityObservation::from_face_scores(bbox, similarity, impostor, margin, None),
                );
            }
            tracklet
        })
        .collect()
}

fn bbox_from_center(cx: f32, cy: f32, width: f32, height: f32) -> BBox {
    BBox {
        x1: cx - width * 0.5,
        y1: cy - height * 0.5,
        x2: cx + width * 0.5,
        y2: cy + height * 0.5,
        confidence: 0.93,
    }
}

fn mean_camera_step_px(keyframes: &[fancam_core::camera::CameraKeyframe]) -> f64 {
    if keyframes.len() < 2 {
        return 0.0;
    }
    let total = keyframes
        .windows(2)
        .map(|window| {
            let dx = window[1].state.cx - window[0].state.cx;
            let dy = window[1].state.cy - window[0].state.cy;
            f64::from(dx.hypot(dy))
        })
        .sum::<f64>();
    total / (keyframes.len() - 1) as f64
}
