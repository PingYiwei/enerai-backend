from app.modules.agents.attachments import detect_image_type


def test_image_type_is_detected_from_bytes_not_request_header() -> None:
    assert detect_image_type(b"\x89PNG\r\n\x1a\ncontent") == "image/png"
    assert detect_image_type(b"\xff\xd8\xffcontent") == "image/jpeg"
    assert detect_image_type(b"RIFF1234WEBPcontent") == "image/webp"
    assert detect_image_type(b"not-an-image") is None
