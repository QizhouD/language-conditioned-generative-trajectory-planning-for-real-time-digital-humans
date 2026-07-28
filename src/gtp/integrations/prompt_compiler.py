"""Convert selected motion intents into bounded Text2Motion prompts."""

from __future__ import annotations

from gtp.schema import MOTION_INTENTS, MotionSegment


def compile_segments(actions: list[int], control_hz: int = 10) -> list[MotionSegment]:
    if not actions:
        return []
    segments: list[MotionSegment] = []
    start = 0
    for index in range(1, len(actions) + 1):
        if index == len(actions) or actions[index] != actions[start]:
            action = MOTION_INTENTS[actions[start]]
            prompt = {
                "idle": "A person stands naturally and remains attentive.",
                "walk": "A person walks forward naturally.",
                "turn": "A person walks while turning smoothly.",
                "stop": "A person slows down and stops naturally.",
                "wave": "A person stops and waves with the right hand.",
            }[action]
            segments.append(MotionSegment(start / control_hz, index / control_hz, action, prompt))
            start = index
    return segments
