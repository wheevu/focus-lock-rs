//! solver — global identity assignment across tracklets.

use std::collections::HashMap;

use crate::camera::{CameraPath, plan_camera_for_identity};
use crate::tracklet::Tracklet;

const TRACKLET_LINK_THRESHOLD: f32 = 0.52;
const TRACKLET_MAX_LINK_GAP: u64 = 540;
const TRACKLET_OVERLAP_PENALTY: f32 = 0.22;

/// Assignment of one tracklet to a global identity id.
#[derive(Debug, Clone, Copy)]
pub struct TrackletAssignment {
    /// Local tracklet id.
    pub tracklet_id: usize,
    /// Global identity id.
    pub identity_id: usize,
    /// Confidence score for this assignment.
    pub confidence: f32,
}

/// Output of global identity solving.
#[derive(Debug, Clone, Default)]
pub struct SolverResult {
    /// Final assignments.
    pub assignments: Vec<TrackletAssignment>,
    /// Chosen identity id for render targeting.
    pub selected_identity_id: Option<usize>,
    /// Camera path produced from assigned tracklets.
    pub camera_path: CameraPath,
}

/// Solve global identity assignments across tracklets and pick the best
/// identity for a camera target.
#[must_use]
pub fn solve(tracklets: &[Tracklet]) -> SolverResult {
    solve_global_assignments(tracklets)
}

#[derive(Debug, Clone, Copy)]
struct TrackletStats {
    tracklet_id: usize,
    first_frame: u64,
    last_frame: u64,
    center_x: f32,
    center_y: f32,
    mean_similarity: f32,
    mean_margin: f32,
    mean_body: f32,
    average_score: f32,
    best_score: f32,
    support: u32,
}

#[derive(Debug, Clone)]
struct IdentityCluster {
    identity_id: usize,
    last_frame: u64,
    center_x: f32,
    center_y: f32,
    mean_similarity: f32,
    mean_margin: f32,
    mean_body: f32,
    average_score: f32,
    best_score: f32,
    support: u32,
}

fn solve_global_assignments(tracklets: &[Tracklet]) -> SolverResult {
    if tracklets.is_empty() {
        return SolverResult::default();
    }

    let mut stats = tracklets
        .iter()
        .filter_map(summarize_tracklet)
        .collect::<Vec<_>>();
    if stats.is_empty() {
        return SolverResult::default();
    }

    stats.sort_unstable_by(|a, b| a.first_frame.cmp(&b.first_frame));

    let mut clusters = Vec::<IdentityCluster>::new();
    let mut assignments = Vec::<TrackletAssignment>::new();

    for stat in stats {
        let best = clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| (idx, tracklet_cluster_score(stat, cluster)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((cluster_idx, score)) = best
            && score >= TRACKLET_LINK_THRESHOLD
        {
            let identity_id = clusters[cluster_idx].identity_id;
            assignments.push(TrackletAssignment {
                tracklet_id: stat.tracklet_id,
                identity_id,
                confidence: score.clamp(0.0, 1.0),
            });
            absorb_tracklet_into_cluster(&mut clusters[cluster_idx], stat);
            continue;
        }

        let identity_id = clusters.len();
        clusters.push(IdentityCluster {
            identity_id,
            last_frame: stat.last_frame,
            center_x: stat.center_x,
            center_y: stat.center_y,
            mean_similarity: stat.mean_similarity,
            mean_margin: stat.mean_margin,
            mean_body: stat.mean_body,
            average_score: stat.average_score,
            best_score: stat.best_score,
            support: stat.support,
        });
        assignments.push(TrackletAssignment {
            tracklet_id: stat.tracklet_id,
            identity_id,
            confidence: 0.62,
        });
    }

    let selected_identity_id = choose_render_identity(&clusters);
    let assignment_rows = assignments
        .iter()
        .map(|assignment| (assignment.tracklet_id, assignment.identity_id))
        .collect::<Vec<_>>();
    let camera_path = selected_identity_id
        .map(|identity_id| plan_camera_for_identity(tracklets, &assignment_rows, identity_id))
        .unwrap_or_default();

    SolverResult {
        assignments,
        selected_identity_id,
        camera_path,
    }
}

fn summarize_tracklet(tracklet: &Tracklet) -> Option<TrackletStats> {
    let first = tracklet.first_frame()?;
    let last = tracklet.last_frame()?;
    if tracklet.is_empty() {
        return None;
    }

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_similarity = 0.0f32;
    let mut sum_margin = 0.0f32;
    let mut sum_body = 0.0f32;
    let mut body_count = 0u32;

    for obs in &tracklet.observations {
        sum_x += obs.bbox.center_x();
        sum_y += obs.bbox.center_y();
        sum_similarity += obs.observation.similarity;
        sum_margin += obs.observation.margin;
        if let Some(body) = obs.observation.body_similarity {
            sum_body += ((body + 1.0) * 0.5).clamp(0.0, 1.0);
            body_count = body_count.saturating_add(1);
        }
    }

    let support = tracklet.len() as u32;
    let support_f = support.max(1) as f32;
    let center_x = sum_x / support_f;
    let center_y = sum_y / support_f;
    let mean_similarity = sum_similarity / support_f;
    let mean_margin = sum_margin / support_f;
    let mean_body = if body_count > 0 {
        sum_body / body_count as f32
    } else {
        0.0
    };

    Some(TrackletStats {
        tracklet_id: tracklet.id,
        first_frame: first,
        last_frame: last,
        center_x,
        center_y,
        mean_similarity,
        mean_margin,
        mean_body,
        average_score: tracklet.average_composite_score(),
        best_score: tracklet.best_composite_score(),
        support,
    })
}

fn tracklet_cluster_score(tracklet: TrackletStats, cluster: &IdentityCluster) -> f32 {
    let appearance = cosine_like(
        tracklet.mean_similarity,
        tracklet.mean_margin,
        cluster.mean_similarity,
        cluster.mean_margin,
    );

    let body = if tracklet.mean_body > 0.0 && cluster.mean_body > 0.0 {
        1.0 - (tracklet.mean_body - cluster.mean_body)
            .abs()
            .clamp(0.0, 1.0)
    } else {
        0.5
    };

    let dx = tracklet.center_x - cluster.center_x;
    let dy = tracklet.center_y - cluster.center_y;
    let distance = dx.hypot(dy);
    let proximity = (1.0 - (distance / 900.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    let temporal = if tracklet.first_frame >= cluster.last_frame {
        let gap = tracklet.first_frame - cluster.last_frame;
        if gap <= TRACKLET_MAX_LINK_GAP {
            (1.0 - (gap as f32 / TRACKLET_MAX_LINK_GAP as f32)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        (1.0 - TRACKLET_OVERLAP_PENALTY).clamp(0.0, 1.0)
    };

    tracklet.average_score.clamp(0.0, 1.0).mul_add(
        0.10,
        ((appearance + 1.0) * 0.5)
            .clamp(0.0, 1.0)
            .mul_add(0.45, body * 0.12)
            + proximity * 0.16
            + temporal * 0.17,
    )
}

fn absorb_tracklet_into_cluster(cluster: &mut IdentityCluster, tracklet: TrackletStats) {
    let left = cluster.support.max(1) as f32;
    let right = tracklet.support.max(1) as f32;
    let denom = left + right;

    cluster.last_frame = cluster.last_frame.max(tracklet.last_frame);
    cluster.center_x = cluster.center_x.mul_add(left, tracklet.center_x * right) / denom;
    cluster.center_y = cluster.center_y.mul_add(left, tracklet.center_y * right) / denom;
    cluster.mean_similarity = cluster
        .mean_similarity
        .mul_add(left, tracklet.mean_similarity * right)
        / denom;
    cluster.mean_margin = cluster
        .mean_margin
        .mul_add(left, tracklet.mean_margin * right)
        / denom;
    cluster.mean_body = cluster.mean_body.mul_add(left, tracklet.mean_body * right) / denom;
    cluster.average_score = cluster
        .average_score
        .mul_add(left, tracklet.average_score * right)
        / denom;
    cluster.best_score = cluster.best_score.max(tracklet.best_score);
    cluster.support = cluster.support.saturating_add(tracklet.support);
}

fn choose_render_identity(clusters: &[IdentityCluster]) -> Option<usize> {
    let mut by_identity = HashMap::<usize, f32>::new();
    for cluster in clusters {
        let support_term = (cluster.support as f32).ln_1p() * 0.08;
        let quality = cluster.best_score.clamp(0.0, 1.0).mul_add(
            0.10,
            cluster.average_score.clamp(0.0, 1.0).mul_add(
                0.17,
                cluster
                    .mean_similarity
                    .mul_add(0.42, cluster.mean_margin.clamp(0.0, 1.0) * 0.31),
            ),
        );
        by_identity.insert(cluster.identity_id, quality + support_term);
    }

    by_identity
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(identity_id, _)| identity_id)
}

fn cosine_like(a_sim: f32, a_margin: f32, b_sim: f32, b_margin: f32) -> f32 {
    let dot = a_sim.mul_add(b_sim, a_margin * b_margin);
    let an = a_sim.hypot(a_margin).max(1e-5);
    let bn = b_sim.hypot(b_margin).max(1e-5);
    (dot / (an * bn)).clamp(-1.0, 1.0)
}
