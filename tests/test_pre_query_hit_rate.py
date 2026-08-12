import unittest
from collections import Counter

from eval.conversation_eval import Conversation, tokenize
from eval.pre_query_hit_rate import (
    build_pqhr_samples,
    pure_topic_predict,
    split_public_users,
    trajectory_ensemble_predict,
)


def make_conversation(session_id: str, started_at: float, content: str) -> Conversation:
    return Conversation(
        session_id=session_id,
        content=content,
        started_at=started_at,
        turns=[content],
        bow=tokenize(content),
    )


class PQHRSplitTest(unittest.TestCase):
    def test_split_public_users_sorts_question_ids_before_taking_validation_slice(self):
        users = {
            f"q{index:03d}": [make_conversation(f"s{index}", float(index), f"topic {index}")]
            for index in range(104, -1, -1)
        }

        validation, test = split_public_users(users, validation_user_count=100)

        self.assertEqual(len(validation), 100)
        self.assertEqual(len(test), 5)
        self.assertEqual(list(validation), [f"q{index:03d}" for index in range(100)])
        self.assertEqual(list(test), [f"q{index:03d}" for index in range(100, 105)])

    def test_split_public_users_rejects_a_partition_without_test_users(self):
        with self.assertRaises(ValueError):
            split_public_users({"only": []}, validation_user_count=1)

class PQHRSampleDisciplineTest(unittest.TestCase):
    def test_target_conversation_never_enters_predictor_candidates(self):
        conversations = [
            make_conversation(
                f"session-{index}",
                float(index),
                f"shared project topic detail {index} " * 30,
            )
            for index in range(12)
        ]

        samples = build_pqhr_samples(conversations, trajectory_window=3)

        self.assertTrue(samples)
        for current, _trajectory, past, truth in samples:
            past_ids = {conversation.session_id for conversation in past}
            self.assertNotIn(current.session_id, past_ids)
            self.assertTrue(truth <= past_ids)

    def test_trajectory_ensemble_preserves_the_topic_rank_one_anchor(self):
        past = [
            make_conversation("auth", 1.0, "auth token refresh login"),
            make_conversation("travel", 2.0, "travel flight hotel"),
            make_conversation("auth-fix", 3.0, "auth login cookie fix"),
        ]
        trajectory = Counter()
        for conversation in past[-2:]:
            trajectory.update(conversation.bow)

        ranked = trajectory_ensemble_predict(2)(trajectory, past, 3)

        self.assertEqual(ranked[0], pure_topic_predict(trajectory, past, 1)[0])
        self.assertEqual(len(ranked), len(set(ranked)))


if __name__ == "__main__":
    unittest.main()
