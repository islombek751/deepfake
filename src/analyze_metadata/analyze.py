import subprocess
import json
from pathlib import Path

def check_video_metadata(video_path):
    """
    Analyzes video metadata using ffprobe to detect possible editing, metadata stripping, 
    or suspicious encoding.

    This function runs the `ffprobe` command-line tool to extract the video’s metadata,
    and then inspects it for missing creation or recording dates, unusual encoder information,
    and mismatched stream durations (e.g. between audio and video).

    Parameters:
        video_path (str or Path): The path to the video file to be analyzed.

    Returns:
        dict: A dictionary containing analysis results:
            - recorded_date_found (bool): Whether a recorded date is present in metadata.
            - encoded_date_found (bool): Whether an encoded or creation date is present.
            - suspicious_encoder (bool): True if the encoder seems non-original (e.g. ffmpeg).
            - encoder_name (str or None): The name of the encoder if available.
            - warnings (list of str): Human-readable warnings about potential issues.
            
            If the file is not found or ffprobe fails, returns a dict with key "error".
    """
    if not Path(video_path).exists():
        return {"error": "Video file not found."}

    # Run ffprobe
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {"error": result.stderr}

    metadata = json.loads(result.stdout)
    format_tags = metadata.get("format", {}).get("tags", {})
    streams = metadata.get("streams", [])

    # Analyze metadata
    analysis = {
        "recorded_date_found": False,
        "encoded_date_found": False,
        "suspicious_encoder": False,
        "encoder_name": None,
        "warnings": [],
    }

    for key in format_tags:
        if "recorded" in key.lower():
            analysis["recorded_date_found"] = True
        if "encoded" in key.lower() or "creation" in key.lower():
            analysis["encoded_date_found"] = True

    encoder = format_tags.get("encoder", "") or format_tags.get("writing_library", "")
    analysis["encoder_name"] = encoder
    if any(tool in encoder.lower() for tool in ["handbrake", "ffmpeg", "lavf", "premiere", "adobe"]):
        analysis["suspicious_encoder"] = True
        analysis["warnings"].append(f"Suspicious encoder detected: {encoder}")

    if len(streams) > 1:
        durations = [float(s.get("duration", 0)) for s in streams if "duration" in s]
        if durations and max(durations) - min(durations) > 1.0:
            analysis["warnings"].append("Mismatched audio/video durations detected — possible editing or re-encoding.")

    if not analysis["recorded_date_found"]:
        analysis["warnings"].append("Recorded date missing — metadata may be stripped.")
    if not analysis["encoded_date_found"]:
        analysis["warnings"].append("Encoded or creation date missing.")

     # --- SUMMARY Qo‘shish ---
    if analysis["suspicious_encoder"] or "missing" in " ".join(analysis["warnings"]).lower():
        summary = "Video may have been edited or metadata stripped."
    else:
        summary = "Video appears authentic based on metadata."

    analysis["summary"] = summary
    return analysis
