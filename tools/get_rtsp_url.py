"""
get_rtsp_url.py
----------------
Most IP cameras (Hikvision, Dahua, Uniview, TP-Link, Reolink, etc.) support
ONVIF, which lets you fetch the RTSP stream URL programmatically instead of
guessing the vendor's URL format. Point this at the camera's ONVIF port
(commonly 80 or 8000) with its admin credentials.

Usage:
    python3 get_rtsp_url.py --ip 192.168.1.50 --port 80 --user admin --password 12345

If ONVIF isn't supported/enabled on the camera, fall back to the vendor's
documented RTSP path pattern, e.g.:
    Hikvision: rtsp://user:pass@IP:554/Streaming/Channels/101
    Dahua:     rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0
    Generic:   rtsp://user:pass@IP:554/stream1
"""

import argparse

from onvif import ONVIFCamera


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
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    url = get_rtsp_url(args.ip, args.port, args.user, args.password)
    print("RTSP stream URL:")
    print(url)
    print("\n(Credentials are usually embedded separately — plug this into")
    print(" FootfallTracker(source=url) or add user:pass@ into the URL yourself.)")