def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def should_flush_final_online(
    *,
    mode: str,
    is_speaking: bool,
    has_buffered_online_frames: bool,
    skip_final_online_flush: bool,
) -> bool:
    return (
        not is_speaking
        and mode == "online"
        and has_buffered_online_frames
        and not skip_final_online_flush
    )
