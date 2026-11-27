//! Zero-dependency, zero-float live telemetry client & parser for `a6000.8b.is` / `cheet.is`
//! (`NVIDIA RTX 6000 Ada Generation` + `Xeon Gold 5412U` AVX-512 sovereign workstation).

use mlx_quant_core::ratio::Ratio;
use mlx_quant_core::ternary::Trit;
use std::process::Command;

/// Default public HTTPS telemetry URL with valid SAN certificate for `49.13.101.247`.
pub const DEFAULT_A6000_STATUS_URL: &str = "https://cheet.is/a6000/api/status";
/// Loopback orchestrator telemetry URL when running directly on `49.13.101.247`.
pub const LOCAL_A6000_STATUS_URL: &str = "http://127.0.0.1:8790/a6000/api/status";

/// Parsed zero-float hardware, VRAM, and fleet snapshot from `a6000-orchestrator`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct A6000StatusSnapshot {
    pub node_ip: String,
    pub gpu_name: String,
    pub vcpu_count: u32,
    pub vram_used_mib: u64,
    pub vram_total_mib: u64,
    pub vram_free_mib: u64,
    pub vram_ratio: Ratio,
    pub gpu_util_pct: u32,
    pub temp_celsius: i32,
    pub power_milliwatts: u64,
    pub load_trit: Trit,
    pub improver_cycles: u64,
    pub improver_speedup: Ratio,
    pub raid160_free_gib: u64,
    pub git160_free_gib: u64,
    pub total_synced_github_repos: u32,
    pub healthy_services: Ratio,
}

fn find_key_value_slice<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let mut pattern = String::with_capacity(key.len() + 2);
    pattern.push('"');
    pattern.push_str(key);
    pattern.push('"');
    let mut search_from = 0usize;
    while let Some(rel) = json[search_from..].find(&pattern) {
        let after_key = search_from + rel + pattern.len();
        let rest = json[after_key..].trim_start();
        if let Some(val_rest) = rest.strip_prefix(':') {
            return Some(val_rest.trim_start());
        }
        search_from = after_key;
    }
    None
}

fn extract_str(json: &str, key: &str) -> Option<String> {
    let val = find_key_value_slice(json, key)?;
    let inner = val.strip_prefix('"')?;
    let mut out = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn extract_i64(json: &str, key: &str) -> Option<i64> {
    let val = find_key_value_slice(json, key)?;
    let mut end = 0usize;
    for (i, ch) in val.char_indices() {
        if (i == 0 && ch == '-') || ch.is_ascii_digit() {
            end = i + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    val[..end].parse::<i64>().ok()
}

fn extract_u64(json: &str, key: &str) -> Option<u64> {
    let v = extract_i64(json, key)?;
    u64::try_from(v).ok()
}

fn parse_ratio_str(s: &str) -> Option<Ratio> {
    let (p_s, q_s) = s.split_once('/')?;
    let p = p_s.trim().parse::<u64>().ok()?;
    let q = q_s.trim().parse::<u64>().ok()?;
    if q == 0 {
        return None;
    }
    Some(Ratio::new(p, q))
}

fn section_from<'a>(json: &'a str, section_key: &str) -> &'a str {
    find_key_value_slice(json, section_key).unwrap_or(json)
}

/// Deterministically parse `/a6000/api/status` JSON using pure integers and `Ratio { p, q }`.
pub fn parse_a6000_status_json(json: &str) -> Option<A6000StatusSnapshot> {
    let gpu_sec = section_from(json, "gpu");
    let improver_sec = section_from(json, "subsystem_f_improver");
    let vlan_sec = section_from(json, "vlan_160tb");

    let node_ip = extract_str(json, "node_ip").unwrap_or_else(|| "49.13.101.247".to_string());
    let gpu_name = extract_str(gpu_sec, "name")?;
    let vcpu_count = extract_u64(gpu_sec, "vcpu_count")? as u32;
    let vram_used_mib = extract_u64(gpu_sec, "vram_used_mib")?;
    let vram_total_mib = extract_u64(gpu_sec, "vram_total_mib")?;
    let vram_free_mib = extract_u64(gpu_sec, "vram_free_mib")?;
    let vram_ratio = extract_str(gpu_sec, "vram_ratio_pq")
        .and_then(|s| parse_ratio_str(&s))
        .unwrap_or_else(|| Ratio::new(vram_used_mib, vram_total_mib.max(1)));
    let gpu_util_pct = extract_u64(gpu_sec, "gpu_util_pct").unwrap_or(0) as u32;
    let temp_celsius = extract_i64(gpu_sec, "temp_celsius").unwrap_or(0) as i32;
    let power_milliwatts = extract_u64(gpu_sec, "power_milliwatts").unwrap_or(0);
    let load_trit = match extract_i64(gpu_sec, "load_trit").unwrap_or(0) {
        x if x > 0 => Trit::Pos,
        x if x < 0 => Trit::Neg,
        _ => Trit::Zero,
    };

    let improver_cycles = extract_u64(improver_sec, "total_cycles").unwrap_or(0);
    let improver_speedup = extract_str(improver_sec, "mean_speedup_pq")
        .and_then(|s| parse_ratio_str(&s))
        .unwrap_or(Ratio::ONE);
    let raid160_free_gib = extract_u64(vlan_sec, "raid160_free_gib").unwrap_or(0);
    let git160_free_gib = extract_u64(vlan_sec, "git160_free_gib").unwrap_or(0);
    let total_synced_github_repos =
        extract_u64(vlan_sec, "total_synced_github_repos").unwrap_or(0) as u32;
    let healthy_services = extract_str(json, "healthy_service_ratio_pq")
        .and_then(|s| parse_ratio_str(&s))
        .unwrap_or(Ratio::ZERO);

    Some(A6000StatusSnapshot {
        node_ip,
        gpu_name,
        vcpu_count,
        vram_used_mib,
        vram_total_mib,
        vram_free_mib,
        vram_ratio,
        gpu_util_pct,
        temp_celsius,
        power_milliwatts,
        load_trit,
        improver_cycles,
        improver_speedup,
        raid160_free_gib,
        git160_free_gib,
        total_synced_github_repos,
        healthy_services,
    })
}

/// Resolve status endpoint URL (`A6000_STATUS_URL` env override, else [`DEFAULT_A6000_STATUS_URL`]).
pub fn resolve_a6000_status_url() -> String {
    std::env::var("A6000_STATUS_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_A6000_STATUS_URL.to_string())
}

/// Fetch and parse live telemetry from `a6000-orchestrator` using verified TLS via system `curl`.
pub fn fetch_a6000_status(url: &str) -> Result<A6000StatusSnapshot, String> {
    let out = Command::new("curl")
        .args(["-fsS", "--max-time", "6", url])
        .output()
        .map_err(|e| format!("failed to run curl for {url}: {e}"))?;
    if !out.status.success() {
        let err_msg = String::from_utf8_lossy(&out.stderr);
        return Err(format!("curl failed for {url}: {}", err_msg.trim()));
    }
    let body =
        String::from_utf8(out.stdout).map_err(|e| format!("non-utf8 telemetry body: {e}"))?;
    parse_a6000_status_json(&body)
        .ok_or_else(|| "failed to parse a6000-orchestrator JSON snapshot".to_string())
}
