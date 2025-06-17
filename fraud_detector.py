from typing import List, Tuple, Dict
from collections import Counter
import numpy as np

JUNK_KEYWORDS = ["nice", "cool", "great", "awesome", "lol", "loved it", "interesting", "Nice post bro!"]

def extract_features(logs, config: Dict) -> Tuple[Dict, List[float], List[str]]:
    mutual_upvotes = 0
    time_deltas = []
    junk_comment_flags = 0
    from_user_counter = Counter()
    comment_texts = []

    prev_time = None
    for act in logs:
        if act.type == "upvote_received" and act.from_user:
            from_user_counter[act.from_user] += 1

        timestamp = act.timestamp
        if prev_time and timestamp:
            delta = abs(np.datetime64(timestamp) - np.datetime64(prev_time)).astype('timedelta64[s]').astype(int)
            time_deltas.append(delta)
        prev_time = timestamp

        if act.type == "comment" and act.content:
            content = act.content.lower()
            if any(word in content for word in JUNK_KEYWORDS):
                junk_comment_flags += 1
            comment_texts.append(content)

    most_common_peer = from_user_counter.most_common(1)[0][1] if from_user_counter else 0
    burst_score = sum(1 for d in time_deltas if d < 60)

    features = {
        "mutual_upvote_count": most_common_peer,
        "karma_bursts": burst_score,
        "junk_comment_count": junk_comment_flags,
        "total_activities": len(logs),
    }

    activity_scores = []
    reasons = []
    for act in logs:
        score = 0.0
        reason = ""
        if act.type == "comment" and act.content:
            if any(jk in act.content.lower() for jk in JUNK_KEYWORDS):
                score += 0.6
                reason = "Low-effort comment flagged as karma bait"
        if act.type == "upvote_received" and act.from_user and from_user_counter[act.from_user] > 2:
            score += 0.8
            reason = "Repeated upvotes from the same peer"
        activity_scores.append(score)
        reasons.append(reason)

    return features, activity_scores, reasons






