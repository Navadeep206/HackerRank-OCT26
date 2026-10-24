"""Evaluation metrics tests; these tests never call an external model."""
import unittest
from code.evaluation.evaluate_predictions import classification_metrics, evaluate


def row(user_id, status, issue="crack"):
    return {
        "user_id": user_id, "image_paths": f"images/{user_id}.jpg",
        "claim_status": status, "issue_type": issue, "object_part": "screen",
        "severity": "medium", "valid_image": "true", "evidence_standard_met": "true",
    }


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics_and_confusion_matrix(self):
        result = classification_metrics(
            ["supported", "supported", "contradicted"],
            ["supported", "contradicted", "contradicted"],
            ("supported", "contradicted", "not_enough_information"),
        )
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3, places=4)
        self.assertEqual(result["confusion_matrix"]["supported"]["contradicted"], 1)
        self.assertEqual(result["per_class"]["supported"]["support"], 2)

    def test_evaluate_matches_rows_by_user_and_image_paths(self):
        references = [row("u1", "supported"), row("u2", "contradicted")]
        predictions = [row("u2", "contradicted"), row("u1", "supported")]
        report = evaluate(predictions, references)
        self.assertEqual(report["matched_rows"], 2)
        self.assertEqual(report["missing_predictions"], [])
        self.assertEqual(report["unexpected_predictions"], [])
        self.assertEqual(report["exact_match"]["claim_status"]["accuracy"], 1.0)
        self.assertEqual(report["errors"], [])

    def test_missing_and_unexpected_rows_are_reported(self):
        report = evaluate([row("u1", "supported")], [row("u1", "supported"), row("u2", "contradicted")])
        self.assertEqual(len(report["missing_predictions"]), 1)
        self.assertEqual(report["matched_rows"], 1)

    def test_duplicate_rows_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            evaluate([row("u1", "supported"), row("u1", "supported")], [row("u1", "supported")])

    def test_empty_metric_set_is_well_defined(self):
        result = classification_metrics([], [], ("supported", "contradicted"))
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
