import os
import unittest

from poc.agent_poc import _smoke_subset
from poc.io_utils import read_rows


class TestSmoke(unittest.TestCase):
    def test_smoke_subset(self) -> None:
        rows = read_rows(
            os.path.join("data", "pkpd_acocella_1984_data.csv")
        )
        subset = _smoke_subset(rows)
        self.assertTrue(len(subset) <= 5)
        self.assertTrue(len(subset) > 0)
        first_id = subset[0].subject_id
        self.assertTrue(all(r.subject_id == first_id for r in subset))


if __name__ == "__main__":
    unittest.main()
