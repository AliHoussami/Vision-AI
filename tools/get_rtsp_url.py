"""
get_rtsp_url.py
----------------
Most IP cameras (Hikvision, Dahua, Uniview, TP-Link, Reolink, etc.) support
ONVIF, which lets you fetch the RTSP stream URL programmatically instead of
guessing the vendor's URL format. Point this at the camera's ONVIF port
(commonly 80 or 8000) with its admin credentials.

Usage:
    python3 get_rtsp_url.py --ip 192.168.1.50 --user admin        # prompts for the password
    python3 get_rtsp_url.py --ip 192.168.1.50 --user admin --password-secret cam-north

Passing --password on the command line is supported but discouraged: it
lands in `ps` output and your shell history. Omit it to be prompted, or
resolve it from ${SECRET:NAME} sources with --password-secret.

If ONVIF isn't supported/enabled on the camera, fall back to the vendor's
documented RTSP path pattern, e.g.:
    Hikvision: rtsp://user:pass@IP:554/Streaming/Channels/101
    Dahua:     rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0
    Generic:   rtsp://user:pass@IP:554/stream1
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onvif import ONVIFCamera

from footfall.secrets import get_secret, redact


def get_rtsp_url(ip: str, port: int, user: str, password: str) -> str:
    cam = ONVIFCamera(ip, port, user, password)
    media_service = cam.create_media_service()
    profiles = media_service.GetProfiles()
    profile_token = profiles[0].token  # first profile = usually main/high-res stream

    stream_setup = {
        "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}},
        "ProfileToken": profile_token,
    }
    uri = media_service.GetStreamUri(stream_setup)
    return uri.Uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover a camera's RTSP URL via ONVIF")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", help="discouraged: visible in ps / history")
    parser.add_argument("--password-secret",
                        help="resolve the password from ${SECRET:NAME} sources")
    args = parser.parse_args()

    if args.password_secret:
        password = get_secret(args.password_secret)
    elif args.password is not None:
        password = args.password
    else:
        password = getpass.getpass(f"Password for {args.user}@{args.ip}: ")

    url = get_rtsp_url(args.ip, args.port, args.user, password)
    print("RTSP stream URL:")
    print(redact(url))
    print("\n(The URL above is credential-masked. Store the real password as")
    print(" ${SECRET:NAME} and reference it from the camera source in site.yaml.)")